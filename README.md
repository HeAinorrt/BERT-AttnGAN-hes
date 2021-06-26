# BERT-AttnGAN-hes
Image generation based on given text descriptions has always been a hot topic in the field of deep learning. with the introduction and development of generation confrontation networks and multi-level semantic encoders, the image resolution and semantic consistency have been greatly improved in recent years. However, there are still some deficiencies in the constraints of the authenticity of details and the semantic consistency of words in the composite image. To solve these problems, this paper proposes a text generation image method which combines BERT coding model and AttnGAN generation confrontation network model and optimizes the loss function. With the help of pre-trained BERT model, the text is encoded at sentence and word level, which makes full use of its excellent text coding and strong generalization ability in NLP task. The spatial attention module is added before the first stage image generation module in the AttnGAN model to improve the semantic consistency of the image generated by the model and the rationality of spatial location matching. The synthetic image obtained by combining the BERT model and the optimized AttnGAN antagonism model is better than the contrast model in the score of relevant indicators, and the overall generation effect is more natural and lifelike.
![Image text](https://github.com/HeAinorrt/BERT-AttnGAN-hes/blob/main/imgfile/netstructure.jpg)
## Dependencies
python 3.6.8

pytorch

In addition, please add the project folder to PYTHONPATH and pip install the following packages:
* nltk
* scikit-image
* python-dateutil
* easydict
* pandas
* torchfile
* pillow
* tqdm

DATA
1. Download the birds image data. Extract them to data/birds/
2. Download the cloth image data. Extract them to data/cloth/

Training
* Pre-train DAMSM models:
	* python pretrain_DAMSM.py
* Train image generate models:
	* python main.py
## Examples generated
![Image text](https://github.com/HeAinorrt/BERT-AttnGAN-hes/blob/main/imgfile/attetion6190027.jpg)
![Image text](https://github.com/HeAinorrt/BERT-AttnGAN-hes/blob/main/imgfile/attmap194141.jpg)
