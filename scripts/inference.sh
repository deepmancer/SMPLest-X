#!/usr/bin/env bash
CKPT_NAME=$1
FILE_NAME=$2

NAME="${FILE_NAME%.*}"
EXT="${FILE_NAME##*.}"

# Check if file is jpg or png
case "$EXT" in
    jpg|jpeg|png)
        ;;
    *)
        echo "Error: Only JPG and PNG files are supported."
        exit 1
        ;;
esac

IMG_PATH=/localhome/aha220/Hairdar/modules/SMPLest-X/demo/images/$NAME
OUTPUT_PATH=/localhome/aha220/Hairdar/modules/SMPLest-X/demo/output_frames/$NAME

# mkdir -p $IMG_PATH
mkdir -p $OUTPUT_PATH

# Copy single image
# cp /localhome/aha220/Hairdar/modules/SMPLest-X/demo/images/$FILE_NAME $IMG_PATH/000001.$EXT

# Set environment variables for OpenGL rendering
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# inference with smplest_x
PYTHONPATH=../:$PYTHONPATH \
python main/inference.py \
    --num_gpus 1 \
    --file_name $NAME \
    --ckpt_name $CKPT_NAME \
    --end 1 \

# Copy result image
# cp $OUTPUT_PATH/000001.$EXT /localhome/aha220/Hairdar/modules/SMPLest-X/demo/result_$FILE_NAME

# rm -rf /localhome/aha220/Hairdar/modules/SMPLest-X/demo/input_frames
# rm -rf /localhome/aha220/Hairdar/modules/SMPLest-X/demo/output_frames
