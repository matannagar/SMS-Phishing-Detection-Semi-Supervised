# SMS-Phishing-Detection-Semi-Supervised
## SpamGAN
This repository was created in order to try and implement a SpamGAN. <br>
A generative adversarial network which relies on limited set of labeled data as well as unlabeled data for opinion spam detection. <br>
<br>
The SpamGAN algorithm offers a way in which we can artificially generate new data to incease our dataset and allow the base algorithm to improve.
This algorithm conssits of a fine-tuned GPT2 model that generates new text and also several trained supervised classification algorithms (as mentioned in the diagram below).

SpamGAN aims to improve the machine-learning algorithms by attending to the issue of limited data in the area of SMS-Phishing. 

## Requirements:
* Transformers
* Pytorch
* Matplotlib

## Steps to run this repository:
* GPT folder holds the code to generating a fine-tuned GPT2 model that can create SMS messages based on our limited dataset.
Refer to the folder and attend readme to generate the model.
* After generating the new model, we can then pass it into the SpamGAN folder.
* Run the classification algorithms in order to generate the pretrained supervised learning models
* Run the Dataset generation file to generate and classify the new data
* Repeat until necessary.

## Project Structure:

<img src="https://raw.githubusercontent.com/matannagar/SMS-Phishing-Detection-Semi-Supervised/master/Repository%20overview.jpg"  width="700" height="500" />

## Teammates ✨

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LiorAtiya"><img src="https://i.ibb.co/G9Nq6X0/Screenshot-2021-12-01-221123.png" width="100px;" alt=""/><br /><sub><b>Lior Atiya</b></sub></a><br /> </td>
    <td align="center"><a href="https://github.com/OfirOvadia96"><img src="https://i.ibb.co/3kXVGdg/Screenshot-2021-12-01-220951.png" width="100px;" alt=""/><br /><sub><b>Ofir Ovadia</b></sub></a><br /> </td>
  </tr>
</table>
