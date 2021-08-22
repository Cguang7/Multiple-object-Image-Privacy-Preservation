# Multiple-object-Image-Privacy-Preservation
This project is aimed to achieve the goal of image privacy preservation.  This project contains two scenarios, the first one is to find objects that are related to people and the scene background in the image, then deal with the area of these objects and relaize the privacy preservation. The second scenario is to classify different importance people in the image, and use different methods to deal with the area of different people.  

In order to achieve our goals, we have referred some excellent programs and made some changes to them. Links to these referenced projects are shown below, and the modified code is shown in this project.


# Usage

# Scenario1: privacy preserving through related objects and areas

We try to combine natural language processing with objective detection in the image to find objects associated with the character and the scene in the image.

1) First, we use objective detection (https://github.com/facebookresearch/detr)to get the bounding box of the person and objects in each image.
   After some modifications, the code file is Scenario1/Find object.ipynb

2) Then, we need get image captions and target words' attention maps. So, we have referred
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

   After some modifications, the code file is Scenario1/caption.py

3) Next, we use Scenario1/Caption_Analysis.ipynb to get the verb related word w1. And we select target attention masks from target words' attention maps that we have got in step1. These masks correspond to word w1.
   The code file Scenario1/Caption_Analysis.ipynb is modified from porject https://github.com/clarkkev/attention-analysis

4) We also use VRD or the distance between persons and objects to reinforce the positioning of objects related to the person. 
   The reference VRD item is https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
   The appropriately modified code file for this project is Scenario1/visualize_custom_SGDet.ipynb. And the code to calculate the distance between bounding boxes in this   project is Scenario1/boxdistance.py 

5) What's more, we try to locate objects that are related to the scene. So, we select target attention masks from target words' attention maps that we have got in step1. These masks correspond to scene related word w2.

   In addition, in order to strengthen the positioning effect of related items, we train a scene classifier and get scene attention masks.
   The VIT model used for train is referred from https://github.com/rwightman/pytorch-image-models

   Then we use an attention explaining method https://github.com/hila-chefer/Transformer-Explainability to visualize the attention that is extracted from the model when classifying the scene.
   The appropriately modified code file for this project is Scenario1/explain.py

6) Last, we use these masks and mosaic/blur/noise technologies to deal images.
   The corresponding code files are in Scenario1/dealimage


# Scenario2: privacy preservation of different charactersin the image

For images containing multiple people, we need to distinguish the priority of these people and adopt different privacy preservation methods for people of different importance.
1) First, we also use objective detection (https://github.com/facebookresearch/detr) to get bounding boxes of all characters in the image, and get different charaters' attention weights in the image.
   After some modifications, the code file in this project is Scenario2/Find person.ipynb

2) Next, we need calculate the attention score and distance score, then calculate the final score according to our method that we have set.
   The corresponding code is also in Scenario2/Find person.ipynb

3) Finally, we use mosaic to deal with the area of senior people, and replace major human face.
   The corresponding code files are in Scenario2/dealimage

   We use https://github.com/podgorskiy/ALAE to genrate the fake face of the major character.
   The appropriately modified code file for this project is ALAE-mytest.py


