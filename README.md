# wpolo-scoresheet-ocr



### Title

Water Polo Scoresheet Reader and Data Organizer Using Optical Character Recognition (OCR)<br/>
Sapienza Università di Roma<br/>
Computer Vision 2024



### Abstract

Every major event of a water polo game, including goals, exclusions, penalties, and timeouts, is recorded onto a scoresheet. For this data to be shared digitally, the coach or another league representative must do the tedious task of manually copying every mark on the scoresheet. It would save time if this upload could be done automatically. In this project I will develop a program that takes as input the image of a scoresheet from a water polo game and outputs a file that lists the information in a tabular format. I will use optical character recognition (OCR) as well as OpenCV. Though the methods of the project will develop themselves more clearly as I develop the program, I know that I will need to create a program that can identify the most meaningful areas of the sheet, identify handwritten numbers and letters, and organize its findings into a easy-to-read tables.



### Introduction

During a water polo game, there is a set of people, designated as the table, that manages the clock, updates the scoreboard with scores and penalties, keeps an official scoresheet, and carries out a few other administrative tasks. The scoresheet is the official record of the game. It includes information about the time and location of the game, the rosters for both teams, and a running record of the time of every major event, including the timestamp and player/team involved for every goal, exclusion, penalty, and timeout. An example can be seen below.

<p align="center">
  <img src="assets/scoresheet.jpeg" width="500">
</p>
At the end of the game, both coaches and the referees must sign the scoresheet to validate its accuracy, and this scoresheet will be transferred to the governing league. When possible, a copy will be given to either coach. Often, especially in leagues with less funding, these scoresheets will be forgotten. In more official leagues, someone will manually type in the data from this sheet so that the league can publish the statistics for every player. The manual input of data is time consuming and tedious, and if done by a coach, takes away time that the coach could be using to analyze previous matches or develop team strategy.
<br/><br/>
Therefore it would be beneficial to the sport if a computer vision system could be created so that a coach or league administrator could take a picture of a scoresheet with a smartphone, and the system could automatically read in the data, organize it, and post it to a website to be available to everyone in the league.



### Literature Review

- [OpenCV](https://opencv.org/)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)



### Dataset
- [Roboflow Dataset - TEAM](https://app.roboflow.com/chris-kreienkamp/water-polo-gamelog-team/3)
- [Roboflow Dataset - REMARKS](https://universe.roboflow.com/chris-kreienkamp/water-polo-gamelog-remarks/dataset/2)
- [PyTorch Dataset - MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
