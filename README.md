# Question-Answer-System
## Introduction
This is a question-answer system project. This project is divided into few parts:
- Simple GPT and BERT examples to compare and choose which one to use
- Question-answer system based on GPT
- Question-answer system based on BERT
- Android app developed based on one question-answer system
## Requirement
- OS: Ubuntu
- Python >= 3.9
- openai
- pytorch >= 1.11
## GPT example
This GPT example could transform natural language into sql instructions.
The result of this example is:
```
The output for "Display the lowest salary from the Worker table" is:
output: Select min(Salary) from Worker where Salary not in (Select min(Salary) from Worker);


The output for "Tell me the count of employees working in the department HR." is:
output: Select count(*) from Worker where DEPARTMENT = 'HR';


The output for "Get salary details of the Workers whose AGE lies between 25 and 35" is:
output: Select Salary from Worker where AGE between 25 and 35;
```
### Usage
To run the example, enter `GPTtry` and run
```
python GPT3.py
```
## BERT example
The BERT example is a question-answer system with only one training data. Provides the description of machine learning:
> Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
Set the question as
> What are machine learning models based on?
We can get this output:
```
Answer: sample data,
Score: 0.8846667408943176
```
### Usage
To runthe example, enter `BERTtry` and run
```
python BERT.py
```

## GPT based question-answer system
The GPT based question-answer system is implemented with the help of [the guide](https://www.twilio.com/blog/ultimate-guide-openai-gpt-3-language-model).
It could answer some simple questions. For example:
```
You: What is python?
GPT-3: Python is a programming language.
You: Do you know how to code?
GPT-3: Yes, I know how to code.
You: How to code?
GPT-3: In Python, the most basic way to code is to use the print statement.
You: Which language should I learn first?
GPT-3: I think it depends on what you want to do with coding. For example, if you want to make websites, you might want to learn HTML and CSS first.
```
### Usage
To run the Question-Answer system, enter `QAsystem-GPT` and run
```
python chat.py
```
## BERT based question-answer system

## Android app

The detailed progress information could be find in `plan`.
