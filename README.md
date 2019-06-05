# Triple-Adjacent-Frame-Generative-Network
Simple Tensorpack implementation of "Triple-Adjacent-Frame Generative Network for Blind Video Motion Deblurring " (TAFDeblurGAN)
# for training
run python train.py

#If you want to continue train, before you train, you should change the START_EPOCH into the breakpoint in the train.py
run python train.py --continue --load /The path of the checkpoints/the name of the last of DATA-00000-OF-00001 file/
# for testing

#the test example is formated by 2×2 images
# |---the previous frame---|---the current frame---|
# |----the last frame------|---the ground truth----|
 
run python test.py  --input_path /PATH/TO/YOUR/TESTING_EXAMPLE/   --output_path /PATH/TO/YOUR/SAVED/IMAGES/
