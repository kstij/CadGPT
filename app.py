import os
from flask import Flask, render_template, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline
import open3d as o3d
import numpy as np
from PIL import Image
import io
import base64
import logging
import time
from werkzeug.serving import run_simple
import gc
import cv2
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS
from point_e.util.point_cloud import PointCloud
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Check for GPU availability
if not torch.cuda.is_available():
    logger.warning("CUDA is not available. Running on CPU will be very slow!")
    logger.warning("Consider using a machine with NVIDIA GPU for better performance.")

# Initialize the models
try:
    logger.info("Loading Stable Diffusion model...")
    # Using a smaller model that requires less memory
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Using float32 for better compatibility
        use_safetensors=False,  # Disable safetensors to reduce memory usage
        low_cpu_mem_usage=True
    )
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    # Initialize Point-E models
    logger.info("Loading Point-E models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the base model
    base_name = 'base40M-textvec'
    base_model = MODEL_CONFIGS[base_name]()
    base_model.eval()
    base_model.to(device)
    base_diffusion = load_checkpoint(base_name, device)
    
    # Load the upsampler model
    upsampler_name = 'upsample'
    upsampler_model = MODEL_CONFIGS[upsampler_name]()
    upsampler_model.eval()
    upsampler_model.to(device)
    upsampler_diffusion = load_checkpoint(upsampler_name, device)
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def estimate_depth_map(image):
    """Estimate depth map from RGB image using edge detection and intensity gradients."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    # Normalize depth map
    depth_map = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    return depth_map

def generate_3d_from_image(image):
    try:
        start_time = time.time()
        logger.info("Starting 3D model generation...")
        
        # Convert image to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Generate depth map
        logger.info("Generating depth map...")
        depth_map = estimate_depth_map(image)
        
        # Create point cloud with depth information
        points = []
        colors = []
        step_size = 4  # Reduced step size for better detail
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # Get depth value
                depth = depth_map[y, x]
                
                # Get color
                color = img_array[y, x] / 255.0
                
                # Calculate 3D position
                x_norm = (x - width/2) / (width/2)
                y_norm = (y - height/2) / (height/2)
                z = depth * 2.0  # Scale depth for better visualization
                
                points.append([x_norm, y_norm, z])
                colors.append(color)
        
        points = np.array(points)
        colors = np.array(colors)
        
        logger.info(f"Point cloud created with {len(points)} points")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals with improved parameters
        logger.info("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Remove outliers
        logger.info("Removing outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Create mesh using Poisson surface reconstruction with improved parameters
        logger.info("Creating mesh...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=True
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Smooth the mesh
        logger.info("Smoothing mesh...")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        end_time = time.time()
        logger.info(f"3D model generation completed in {end_time - start_time:.2f} seconds")
        
        return mesh
    except Exception as e:
        logger.error(f"Error in 3D generation: {str(e)}")
        raise

def generate_3d_with_point_e(image):
    try:
        start_time = time.time()
        logger.info("Starting Point-E 3D model generation...")
        
        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Generate base point cloud
        logger.info("Generating base point cloud...")
        base_pc = base_diffusion.sample(
            base_model,
            image_tensor,
            guidance_scale=3.0,
            num_inference_steps=20,
        )
        
        # Upsample the point cloud
        logger.info("Upsampling point cloud...")
        upsampled_pc = upsampler_diffusion.sample(
            upsampler_model,
            base_pc,
            guidance_scale=3.0,
            num_inference_steps=20,
        )
        
        # Convert to Open3D point cloud
        points = upsampled_pc.coords.cpu().numpy()
        colors = upsampled_pc.channels['RGB'].cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create mesh using Poisson surface reconstruction
        logger.info("Creating mesh...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=True
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Smooth the mesh
        logger.info("Smoothing mesh...")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        end_time = time.time()
        logger.info(f"Point-E 3D model generation completed in {end_time - start_time:.2f} seconds")
        
        return mesh
    except Exception as e:
        logger.error(f"Error in Point-E 3D generation: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        prompt = request.json.get('prompt')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        logger.info(f"Generating model for prompt: {prompt}")
        start_time = time.time()
        
        # Generate image from prompt with fewer steps
        logger.info("Generating image from prompt...")
        image = pipe(
            prompt,
            num_inference_steps=15,
            guidance_scale=7.0
        ).images[0]
        logger.info("Image generated successfully")
        
        # Clear memory after image generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate 3D model using Point-E
        mesh = generate_3d_with_point_e(image)
        
        # Save the mesh to a temporary file
        temp_path = "temp_model.obj"
        o3d.io.write_triangle_mesh(temp_path, mesh)
        
        # Read the file and encode it
        with open(temp_path, 'rb') as f:
            model_data = f.read()
        
        # Clean up
        os.remove(temp_path)
        
        # Clear memory again
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        end_time = time.time()
        logger.info(f"Total generation time: {end_time - start_time:.2f} seconds")
        
        # Return the model data
        return jsonify({
            'model': base64.b64encode(model_data).decode('utf-8'),
            'generation_time': f"{end_time - start_time:.2f} seconds"
        })
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use werkzeug's run_simple for better server handling
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True, use_evalex=True) 