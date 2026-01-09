<h2>TensorFlow-FlexUNet-Image-Segmentation-High-Resolution-Concrete-Damage (2026/01/10)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for HRCDS 
based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and <a href="https://drive.google.com/file/d/1KDbqWr8PLj4oPLHy1QhHSXDQUiFXEkfK/view?usp=sharing">Augmented-HRCDS-ImageMask-Dataset.zip </a> which was derived by us from<br><br>

<a href="https://data.mendeley.com/datasets/6x4dzzrs2h/2">MDMCS: A Benchmark Dataset for Multi-Damage Monitoring of Concrete Structures</a>
<br><br>

<hr>
<b>Actual Image Segmentation for HRCDS Images of 1080x720 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks, but they lack precision in certain areas. <br><br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10042.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10042.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10042.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10182.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10182.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10297.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10297.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10297.png" width="320" height="auto"></td>
</tr>
<!--
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/img_0028.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/img_0028.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/img_0028.png" width="320" height="auto"></td>
</tr>
 -->
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from the following google drive:<br><br>
<a href="https://data.mendeley.com/datasets/6x4dzzrs2h/2">MDMCS: A Benchmark Dataset for Multi-Damage Monitoring of Concrete Structures</a>
<br><br>
<b>Description</b><br>
Concrete structures deteriorate over time due to environmental exposure and mechanical stress, leading to various types of damage such as cracking, 
spalling, corrosion, and exposed rebar.<br>
 Automated detection using deep learning-based computer vision techniques is limited by the lack of high-quality, annotated datasets. <br>
 To address this challenge, this paper presents MDMCS (Multi-Damage Monitoring of Concrete Structures), a dataset of 1,200 images with 
 precise pixel-wise annotations involving four types of damage (cracking, spalling, corrosion, and exposed rebar) and 
 diverse lighting conditions and material textures. <br>
 The dataset has been evaluated using six state-of-the-art segmentation models, 
 validating the efficacy of the dataset and providing benchmarks for damage detection models. 
 MDMCS will facilitate advances in artificial intelligence-powered structural monitoring and robot-assisted automatic inspection for
  improving the operation and maintenance of concrete structures.
<br><br>
<b>Citation</b><br>
guo, pengwei; Bao, Yi (2025), <br>
“MDMCS: A Benchmark Dataset for Multi-Damage Monitoring of Concrete Structures”, <br>
Mendeley Data, V2, doi: 10.17632/6x4dzzrs2h.2
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
<br>
<br>
<h3>
2 HRCDS ImageMask Dataset
</h3>
<h4>2.1  Download Augmented Dataset</h4>
 If you would like to train this HRCDS Segmentation model by yourself,
 please download the original dataset from <a href="https://drive.google.com/file/d/1KDbqWr8PLj4oPLHy1QhHSXDQUiFXEkfK/view?usp=sharing">
 Augmented-HRCDS-ImageMask-Dataset.zip </a>, expand the downloaded, and put it  under <b>./dataset </b> folder  to be.<br>
<pre>
./dataset
└─HRCDS
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>HRCDS Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/HRCDS/HRCDS_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br><br>
<h4>2.2 Augmented Dataset Derivation </h4>
The foloder struture of the original HRCDS is the following.<br>
<pre>
./HRCDS
├─test_annotations
├─test_image
├─test_mark_color
├─test_mask
├─train_annotations
├─train_image
├─train_mask
├─val_annotations
├─val_image
└─val_mask
</pre>
At first, we generated a  <b>HRCDS-master</b> dataset by merging image and annotations of test_*, train_*, and valid_*  subsets.<br>
<pre>
./HRCDS-master
├─image
│      ├─test_0001.jpg
 ...
│      ├─train_0001.jpg
 ...   
│      └─val_0100.jpg
└─annotations
       ├─test_0001.json
       ...
       ├─train_0001.json
       ...   
       └─val_0100.json
</pre> 
<br>
Then we generated  our Augmented Image Mask Dataset from 1,200 JPG files in <b>HRCDS-master/image</b> and 1,200 JSON files in <b>HRCDS-master/annotations</b> by using the following 2 Python scripts.<br>
<ul>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
</ul>
We also used the following color-class mapping table to generate colorized masks, and define a rgb_map mask format between indexed colors and rgb colors.<br>
<br>
<a id="color-class-mapping-table"><b>HRCDS color class mapping table</b></a>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Crack.png' widith='40' height='25'</td><td>(255, 0, 0)</td><td>Crack</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/ExposedRebar.png' widith='40' height='25'</td><td>(0, 255, 0)</td><td>ExposedRebar</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Corrosion.png' widith='40' height='25'</td><td>(0, 255, 255)</td><td>Corrosion</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Spalling.png' widith='40' height='25'</td><td>(128, 128, 128)</td><td>Spalling</td></tr>
</table>
<br>
<h4>2.3 Train Dataset Samples </h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained HRCDS TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/HRCDS/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/HRCDS and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowFlexUNet.py">TensorflowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 5

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for HRCDS 1+4 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;HRCDS 1+4
rgb_map = {(0,0,0):0, (0,255,255):1, (255,0,0):2, (153,76,0):3, (0,153,0):4}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middle point (35,36,37,38)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (70,71,72,73)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 73 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/train_console_output_at_epoch73.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/HRCDS/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/HRCDS/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HRCDS</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for HRCDS.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/evaluate_console_output_at_epoch73.png" width="880" height="auto">
<br><br>Image-Segmentation-HRCDS

<a href="./projects/TensorFlowFlexUNet/HRCDS/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this HRCDS/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.086
dice_coef_multiclass,0.9745
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HRCDS</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for HRCDS.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/HRCDS/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for HRCDS Images of  1080x720 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10105.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10105.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10105.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10255.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10255.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10255.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10388.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10388.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10388.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10557.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10557.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10557.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10744.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10744.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10744.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/images/10488.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test/masks/10488.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRCDS/mini_test_output/10488.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Crack</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Crack">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Image-Segmentation-Crack
</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Concrete-Crack
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
