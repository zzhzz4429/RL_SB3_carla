#import carla
import numpy as np
import torch
from PIL import Image
from model import LidarCenterNet
from config import GlobalConfig
from data import lidar_to_histogram_features, draw_target_point
import os
import glob
import sys
# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
# Specify the relative path to the CARLA directory from the current directory
relative_path = '../carla/PythonAPI/carla/'

# Get the absolute path to the CARLA directory
carla_directory = os.path.abspath(relative_path)

try:
    egg_file = glob.glob(os.path.join(carla_directory, 'dist', 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )))[0]
    sys.path.append(egg_file)
except IndexError:
    print("CARLA egg file not found in the specified directory.")
    print("Searched directory:", carla_directory)
# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Set up the TransFuser model
config = GlobalConfig()
transfuser = LidarCenterNet(config, 'cuda', 'transFuser')
transfuser.load_state_dict(torch.load('~/transfuser/model_ckpt/models_2022/transfuser/model_seed1_39.pth'))
transfuser.eval()

# Spawn the ego vehicle
spawn_point = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))
ego_vehicle = world.spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'), spawn_point)

# Attach RGB camera sensor
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Attach LiDAR sensor
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels', '64')
lidar_bp.set_attribute('points_per_second', '100000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_bp.set_attribute('range', '50')
lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)

# Attach GPS sensor
gps_bp = world.get_blueprint_library().find('sensor.other.gnss')
gps_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
gps = world.spawn_actor(gps_bp, gps_transform, attach_to=ego_vehicle)

# Main loop
try:
    while True:
        # Tick the simulation
        world.tick()

        # Get the sensor data
        camera_data = camera.listen(lambda image: Image.frombytes(mode='RGB', size=(image.width, image.height), data=image.raw_data))
        lidar_data = lidar.listen(lambda point_cloud: np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')).reshape([-1, 3]))
        gps_data = gps.listen(lambda data: (data.latitude, data.longitude))

        # Prepare the input data for the model
        image = np.array(camera_data)
        lidar_bev = lidar_to_histogram_features(lidar_data)
        target_point = np.array([0.0, 0.0])  # Dummy target point
        target_point_image = draw_target_point(target_point)
        velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y, ego_vehicle.get_velocity().z])

        # Feed the data through the TransFuser model
        with torch.no_grad():
            fused_features = transfuser._model.forward_features(torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().cuda(),
                                                                 torch.from_numpy(lidar_bev).unsqueeze(0).float().cuda(),
                                                                 torch.from_numpy(velocity).unsqueeze(0).float().cuda())

        # Print the shape of the fused features
        print("Fused features shape:", fused_features.shape)

finally:
    # Destroy the actors
    camera.destroy()
    lidar.destroy()
    gps.destroy()
    ego_vehicle.destroy()