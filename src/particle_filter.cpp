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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Set up normal distributions
	std::random_device device;
	std::mt19937 generator(device());
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	particles.clear();
	weights.clear();

	for (int i = 0; i < NUM_PARTICLES; i++) {
		Particle new_particle;

		new_particle.id = i;
		new_particle.weight = 1;

		// Sample from our normal distributions
		new_particle.x = dist_x(generator);
		new_particle.y = dist_y(generator);
		new_particle.theta = dist_theta(generator);

		// Add new particle to particle list, and update the weights list
		particles.push_back(new_particle);
		weights.push_back(new_particle.weight);
	}

	// Update our num_particles count
	num_particles = particles.size();

	// We're done with initialization
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double pred_x = 0.0;
	double pred_y = 0.0;
	double pred_theta = 0.0;

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	std::default_random_engine generator;
	// std::random_device device;
	// std::mt19937 generator(device());
	std::normal_distribution<double> noise_x(0.0, std_x);
	std::normal_distribution<double> noise_y(0.0, std_y);
	std::normal_distribution<double> noise_theta(0.0, std_theta);

	for (int i = 0; i < particles.size(); i++) {		
		// Predict the new measurements based on the bicycle motion model
		if (yaw_rate != 0) {
			// Non-zero yaw rate means we have to factor in rotation
			pred_x = particles[i].x + ((velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)));
			pred_y = particles[i].y + ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))));
			pred_theta = particles[i].theta + (yaw_rate * delta_t);
		} else {
			// Else, just continue moving in the current direction at the current speed
			pred_x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
			pred_y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
			pred_theta = particles[i].theta;
		}

		// Update the particle with our predicted (noisy) measurements
		particles[i].x = pred_x + noise_x(generator);
		particles[i].y = pred_y + noise_y(generator);
		particles[i].theta = pred_theta + noise_theta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks_within_range, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double distance = 0.0;
	auto closest_index = -1;
	double closest_dist = -1;

	// For each observation
	for (int i = 0; i < observations.size(); i++) {

		// Reset the closest index each time
		closest_index = -1;
		closest_dist = -1;

		// For each predicted landmark
		for (int j = 0; j < landmarks_within_range.size(); j++) {

			// Calculate the euclidean distance between the two
			distance = dist(observations[i].x, observations[i].y, landmarks_within_range[j].x, landmarks_within_range[j].y);
			
			// If the calculated distance is less than our last calculated closest distance, update it.
			if (distance < closest_dist || closest_dist == -1 ) {
				closest_index = j;
				closest_dist = distance;
			}
		}

		// By this point we found the closest id. Assign it to our prediction
		observations[i].id = closest_index;

		// std::cout << "[" << i << "] Matching landmark: " << closest_index << ", lm coords: " << landmark_list[closest_index].x_f << ", " << landmark_list[closest_index].y_f << std::endl;

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
	
	long double mgp = 0.0;
	long double calculated_weight = 1.0;
	double exp_power, x, ux, y, uy, left_part, right_part;
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	int corresponding_landmark_index = 0;
	double dist_from_landmark = 0.0;
	
	std::vector<LandmarkObs> transformed_observations;
	std::vector<LandmarkObs> landmarks_within_range;
	LandmarkObs landmark_within_range;

	num_particles = particles.size();

	// Clear all weights as we'll be recalculating them.
	weights.clear();

	// For each particle
	for (int i = 0; i < num_particles; i++) {

		// Assume the current particle is the 'Vehicle'
		// We'll use this particle's position and rotation.

		// For performance reasons, we only want to work with
		// Landmarks that are within the sensor range of the 'Vehicle'
		landmarks_within_range.clear();
		for (auto l : map_landmarks.landmark_list) {
			dist_from_landmark = dist(l.x_f, l.y_f, particles[i].x, particles[i].y);
			if (dist_from_landmark <= sensor_range) {
				landmark_within_range.x = l.x_f;
				landmark_within_range.y = l.y_f;
				landmark_within_range.id = l.id_i;
				landmarks_within_range.push_back(landmark_within_range);
			}
		}

		// Translate the observations that this particle saw to
		// their inferred positions on the map (relative to the particle)

		// Clear the old list for a new set of translations
		transformed_observations.clear();

		// Translate observations into map coordinates
		// std::cout << "Particle (" << i << "): " << particles[i].x << ", " << particles[i].y << ", " << particles[i].theta << ", " << std::endl;
		for (int j = 0; j < observations.size(); j++) {
			translation_x = particles[i].x;
			translation_y = particles[i].y;
			rotation_theta = particles[i].theta;

			// LandmarkObs translated_obs = observations[j];
			LandmarkObs translated_obs;

			translated_obs.x = (observations[j].x * cos(rotation_theta)) - (observations[j].y * sin(rotation_theta)) + translation_x;
			translated_obs.y = (observations[j].x * sin(rotation_theta)) + (observations[j].y * cos(rotation_theta)) + translation_y;
			transformed_observations.push_back(translated_obs);

			// std::cout << "[" << j << "] Obs-orginal: " << observations[j].x << ", " << observations[j].y <<  "Obs-trans: " << translated_obs.x << ", " << translated_obs.y << std::endl;
		}

		// Associate translated observations with the observations
		// std::cout << "Transformation (Before)";
		// std::cout << transformed_observations[0].id << std::endl;
		dataAssociation(landmarks_within_range, transformed_observations);
		// std::cout << "Transformation (After)";
		// std::cout << transformed_observations[0].id << std::endl;

		// Calculate multivariate gaussian probability for each observation
		calculated_weight = 1.0;
		for (int k = 0; k < transformed_observations.size(); k++) {

			x = transformed_observations[k].x;
			y = transformed_observations[k].y;

			corresponding_landmark_index = transformed_observations[k].id;

			ux = landmarks_within_range[corresponding_landmark_index].x;
			uy = landmarks_within_range[corresponding_landmark_index].y;

			left_part = ((x - ux) * (x - ux)) / (2. * sigma_x * sigma_x);
			right_part = ((y - uy) * (y - uy)) / (2. * sigma_y * sigma_y);
			exp_power = -1. * (left_part + right_part);
			mgp = (1. / (2. * M_PI * sigma_x * sigma_y)) * exp(exp_power);					

			calculated_weight *= mgp;

			// if (mgp == 0) {
			// 	std::cout << "mgp is ZERO: DEBUG" << std::endl;
			// 	std::cout << "sigma_x:" << sigma_x << std::endl;
			// 	std::cout << "sigma_y:" << sigma_y << std::endl;
			// 	std::cout << "x:" << x << std::endl;
			// 	std::cout << "y:" << y << std::endl;
			// 	std::cout << "ux:" << ux << std::endl;
			// 	std::cout << "uy:" << uy << std::endl;
			// 	std::cout << "x - ux:" << x - ux << std::endl;
			// 	std::cout << "y - uy:" << x - uy << std::endl;
			// 	std::cout << "exp_power" << exp_power << std::endl;
			// 	std::cout << "exp(exp_power)" << exp(exp_power) << std::endl;
			// }

			// std::cout << "mgp: " << mgp << ", Calculated weight:" << calculated_weight << std::endl;

		}

		// std::cout << "Final Weight(" << i << "): " << calculated_weight << endl;
		// if (transformed_observations.size() > 0) 
		particles[i].weight = calculated_weight;
		weights.push_back(calculated_weight);

	}

}

void ParticleFilter::resample() {
	// long double total_weight = 0.0;

	// weights.clear();

	// // std::cout << "Resampling!" << std::endl;
	// // Rebuild list of particle weights, ordered by particle index

	// // std::cout << "Original Weights: {";	
	// for (int i = 0; i < num_particles; i++) {
	// 	total_weight += particles[i].weight;
	// 	weights.push_back(particles[i].weight);
	// 	// std::cout << particles[i].weight << ", ";
	// }
	// // std::cout << "}" << std::endl;

	// std::cout << "Normalized Weights: {";
	// for (int i = 0; i < num_particles; i++) {
	// 	weights[i] = weights[i] / total_weight;
	// 	std::cout << weights[i] << ", ";
	// }
	// std::cout << "}" << std::endl;


	// Prepare to resample
	std::random_device device;
	std::mt19937 generator(device());
	std::discrete_distribution<> weight_dist(weights.begin(), weights.end());

	//Resample the particles
	std::vector<Particle> new_particles;
	new_particles.clear();




	// std::cout << "Original particles: {";
	// for (int i = 0; i < num_particles; i++) {
	// 	std::cout << particles[i].id << ", ";
	// }
	// std::cout << "}" << std::endl;





	// std:cout << "Num particles: " << num_particles << std::endl;
	// std::cout << "new_particles size(BEFORE): " << new_particles.size() << std::endl;
	for (int i = 0; i < num_particles; i++) {
		int sample_index = weight_dist(generator);
		// std::cout << sample_index << ", ";
		new_particles.push_back(particles[sample_index]);
	}
	// std::cout << "new_particles size(AFTER): " << new_particles.size() << std::endl;
	// std::cout << "particles size (BEFORE): " << particles.size() << std::endl;
	particles.clear();
	particles = new_particles;
	// std::cout << "particles size (AFTER): " << particles.size() << std::endl;


	// std::cout << "Resampled particles: {";
	// for (int i = 0; i < num_particles; i++) {
	// 	std::cout << particles[i].id << ", ";
	// }
	// std::cout << "}\n\n" << std::endl;
	
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

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}