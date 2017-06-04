/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES 1000

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Set up normal distributions
	default_random_engine generator;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < NUM_PARTICLES; i++) {
		Particle new_particle;

		new_particle.id = i;
		new_particle.weight = 1;

		// Sample from our normal distributions
		new_particle.x = dist_x(generator);
		new_particle.y = dist_y(generator);
		new_particle.theta = dist_theta(generator);

		// Add new particle to particle list
		particles.push_back(new_particle);
	}

	// Update our num_particles count
	num_particles = particles.size();

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	double pred_x = 0.0;
	double pred_y = 0.0;
	double pred_theta = 0.0;

	default_random_engine generator;
	num_particles = particles.size();

	for (int i = 0; i < num_particles; i++) {
		// Predict the new measurements based on the bicycle motion model
		pred_x = particles[i].x + ((velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)));
		pred_y = particles[i].y + ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))));
		pred_theta = particles[i].theta + (yaw_rate * delta_t);

		// Add noise
		normal_distribution<double> dist_x(pred_x, std_x);
		normal_distribution<double> dist_y(pred_y, std_y);
		normal_distribution<double> dist_theta(pred_theta, std_theta);

		// Update the particle with our predicted (noisy) measurements
		particles[i].x = dist_x(generator);
		particles[i].y = dist_y(generator);
		particles[i].theta = dist_theta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> landmark_list, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double distance = 0.0;

	auto closest_id = -1;
	double closest_dist = -1;

	// For each observation
	for (int i = 0; i < observations.size(); i++) {

		// For each predicted landmark
		for (int j = 0; j < landmark_list.size(); j++) {

			// Calculate the euclidean distance between the two
			distance = dist(observations[i].x, observations[i].y, landmark_list[j].x_f, landmark_list[j].y_f);
			
			if (closest_dist < 0 || closest_dist > distance) {
				closest_id = landmark_list[j].id_i;
				closest_dist = distance;
			}
		}

		// By this point we found the closest id. Assign it to our prediction
		observations[i].id = closest_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double translation_x = 0.0;
	double translation_y = 0.0;
	double rotation_theta = 0.0;
	
	double mgp = 0.0;
	double weight = 1.0;
	double exp_power, x, ux, y, uy;
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
	
	std::vector<LandmarkObs> transformed_observations;

	num_particles = particles.size();

	// For each particle
	for (int i = 0; i < num_particles; i++) {

		// Assume the current particle is the 'Vehicle'
		// We'll use this particle's position and rotation.

		// We first translate the observations that this particle saw to
		// their inferred positions on the map (relative to the particle)

		// Clear the old list for a new set of translations
		transformed_observations.clear();

		// Translate observations into map coordinates
		for (int j = 0; j < observations.size(); j++) {
			translation_x = particles[i].x - observations[j].x;
			translation_y = particles[i].y - observations[j].y;
			rotation_theta = particles[i].theta;

			LandmarkObs translated_obs = observations[j];			
			translated_obs.x = translated_obs.x * cos(rotation_theta) - translated_obs.y * sin(rotation_theta) + translation_x;
			translated_obs.y = translated_obs.x * sin(rotation_theta) + translated_obs.y * cos(rotation_theta) + translation_y;
			transformed_observations.push_back(translated_obs);
		}

		// Associate translated observations with the observations
		dataAssociation(map_landmarks.landmark_list, transformed_observations);

		// Calculate multivariate gaussian probability for each observation
		for (int k = 0; k < transformed_observations.size(); k++) {
			for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
				if (map_landmarks.landmark_list[l].id_i == transformed_observations[k].id) {
					
					x = transformed_observations[k].x;
					y = transformed_observations[k].y;
					ux = map_landmarks.landmark_list[l].x_f;
					uy = map_landmarks.landmark_list[l].y_f;

					exp_power = -((pow(x - ux, 2) / (2 * pow(sigma_x, 2))) + (pow(y - uy, 2) / (2 * pow(sigma_y, 2))));
					mgp = 1. / (2 * M_PI * sigma_x * sigma_y) * exp(exp_power);					

					weight *= mgp;

					break;
				}
			}

		}

		particles[i].weight = weight;

	}

}

void ParticleFilter::resample() {
	// Rebuild list of particle weights, ordered by particle index
	weights.clear();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	default_random_engine generator;
	discrete_distribution<int> weight_dist(weights.begin(), weights.end());

	//Resample the particles
	std::vector<Particle> old_particles;
	old_particles.swap(particles);

	for (int i = 0; i < num_particles; i++) {
		particles.push_back(old_particles[weight_dist(generator)]);
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
