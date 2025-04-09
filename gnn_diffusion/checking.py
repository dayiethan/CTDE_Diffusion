import torch
import numpy as np
from utils import Normalizer, set_seed
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Parameters
n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 10 # horizon, length of each trajectory

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0) 

# Loading training trajectories
all_points1 = []    # want modes 1, 2, 4, 6
all_points2 = []    # want modes 1, 2, 3, 5
with open('data/trajs_noise1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x1, y1 = float(row[4]), float(row[5])
        x2, y2 = float(row[7]), float(row[8])
        all_points1.append([x1, y1])
        all_points2.append([x2, y2])

num_trajectories = 10000
points_per_trajectory = 10

expert_data1 = [
    all_points1[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory1 = expert_data1[0]
x1 = [point[0] for point in first_trajectory1]
y1 = [point[1] for point in first_trajectory1]

expert_data2 = [
    all_points2[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory2 = expert_data2[0]
x2 = [point[0] for point in first_trajectory2]
y2 = [point[1] for point in first_trajectory2]


expert_data1 = np.array(expert_data1)
expert_data2 = np.array(expert_data2)



# Unspliced trajectories to get final positions
orig1 = [
    all_points1[i * 100:(i + 1) * 100]
    for i in range(1000)
]
orig2 = [
    all_points2[i * 100:(i + 1) * 100]
    for i in range(1000)
]
orig1 = np.array(orig1)
orig2 = np.array(orig2)

# print(orig1.shape)
# print(orig2.shape)

# # plot the original trajectories
# plt.figure(figsize=(10, 5))
# for i in range(orig1.shape[0]):
#     plt.plot(orig1[i, :, 0], orig1[i, :, 1], 'b-')
#     plt.plot(orig2[i, :, 0], orig2[i, :, 1], 'r-')

# plt.savefig("figs/original_trajs.png")   

print(expert_data1.shape)
print(expert_data2.shape)

# scatter plots for first points of each trajectory in expert_data1 and expert_data2

plt.figure(figsize=(10, 5))
plt.scatter(expert_data1[:, 0, 0], expert_data1[:, 0, 1], color='blue', label='Expert Data 1')
plt.scatter(expert_data2[:, 0, 0], expert_data2[:, 0, 1], color='red', label='Expert Data 2')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot of First Points in Expert Data')
plt.legend()
plt.savefig("figs/scatter_first_points.png")

