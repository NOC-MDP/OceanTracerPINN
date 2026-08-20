import imageio
import glob
import os

def write_video(glob2,video_path):
    # Grab all matching files and sort them (they'll sort correctly as strings given the YYYY-MM format)
    image_files = sorted(glob.glob(glob2))

    # Write to video
    with imageio.get_writer(video_path, fps=4) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Video created from {len(image_files)} frames.")

# write_video(glob2="infer_output/FWC/FWC_????-??_90_50_180_-180.png",video_path="FWC_animation.mp4")
#write_video(glob2="/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/model_10m_map_????_??.png",video_path="d18O_10m_animation.mp4")
#write_video(glob2="/gws/ssde/j25a/nemo/vol4/thopri/PINN/infer_output/model_zonal_mean_????_??.png",video_path="zonal_animation.mp4")

# write_video(glob2="infer_output/FWC/FWC_????-??_90_50_180_-180.png",video_path="FWC_animation.mp4")
# write_video(glob2="/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output/model_10m_map_????_??.png",video_path="Ba_10m_animation.mp4")
# write_video(glob2="/gws/ssde/j25a/nemo/vol4/thopri/PINN_BARIUM/infer_output/model_zonal_mean_????_??.png",video_path="Ba_zonal_animation.mp4")

write_video(glob2="outputs/inference/images/model_10m_map_????_??.png",video_path="outputs/inference/videos/d18O_10m_animation_TOPAZ.mp4")
write_video(glob2="outputs/inference/images/model_zonal_mean_????_??.png",video_path="outputs/inference/videos/d180_10m_zonal_animation_TOPAZ.mp4")
