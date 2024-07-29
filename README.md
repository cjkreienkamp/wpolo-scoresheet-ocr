# wpolo-scoresheet-ocr
Takes as input a scoresheet from a water polo game and outputs a file that lists the information in a tabular digital format. Uses optical character recognition (OCR) and OpenCV to read in the data.



### Title and Author Information

Water Polo Scoresheet Reader and Data Organizer Using Optical Character Recognition (OCR)
Sapienza Università di Roma
Computer Vision 2024
Christopher Kreienkamp



### Abstract

A short (usually around 250-400 words) description of the paper. Should include what the purpose of the paper is (including the basic research question/problem), the basic design of your project, and the major findings.
To keep records of a water polo game, typically
Introduce the topic. ...
State the problem addressed by the research. ...
Summarize why this problem exists. ...
Explain how the research question was addressed. ...
What were the findings of the research conducted? ...
What is the meaning or impact of your research?



### Introduction

During a water polo game, there is a set of people, designated as the table, that manages the clock, updates the scoreboard with scores and penalties, keeps an official scoresheet, and carries out a few other administrative tasks. The scoresheet is the official record of the game. It includes information about the time and location of the game, the rosters for both teams, and a running record of the time of every major event, including the timestamp and player/team involved for every goal, exclusion, penalty, and timeout. An example can be seen below.

![alt text]([http://url/to/img.png](https://github.com/cjkreienkamp/wpolo-scoresheet-ocr/blob/f4ee7a54a98551f87a549c90c5a0e48c69ce9ae1/assets/scoresheet.jpeg))

At the end of the game, both coaches and the referees must sign the scoresheet to validate its accuracy, and this scoresheet will be transferred to the governing league. When possible, a copy will be given to either coach. Often, especially in leagues with less funding, these scoresheets will be forgotten. In more official leagues, someone will manually type in the data from this sheet so that the league can publish the statistics for every player. The manual input of data is time consuming and tedious, and if done by a coach, takes away time that the coach could be using to analyze previous matches or develop team strategy.

Therefore it would be beneficial to the sport if a computer vision system could be created so that a coach or league administrator could take a picture of a scoresheet with an smartphone, and the system could automatically read in the data, organize it, and post it to a website to be available to everyone in the league.



### Literature Review

- [OpenCV](https://opencv.org/)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [Credit card OCR with OpenCV and Python](https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/)



### Methods:

## Results:

# Discussion:

## Conclusion:

### References and Appendices:
