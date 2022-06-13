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
- transformers
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
The BERT example is a question-answer system with only one training data. 
It is implemented with the help of [the guide](https://towardsdatascience.com/question-answering-with-pretrained-transformers-using-pytorch-c3e7a44b4012) Provides the description of machine learning:
> Machine learning (ML) is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

Set the question as

> What are machine learning models based on?

We can get this output:
```
Answer: sample data,
Score: 0.8846667408943176
```
### Usage
To run the example, enter `BERTtry` and run
```
python BERT.py
```

## GPT based question-answer system
The GPT based question-answer system is implemented with the help of [this guide](https://www.twilio.com/blog/ultimate-guide-openai-gpt-3-language-model).
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
The BERT based question-answer system is implemented with the help of [this guide](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626).
We use an already available fine-tund BERT model from the Hugging Face Transformers library to 
answer questions based on the stories from the CoQA dataset.
### Usage
To run the Question-Answer system, enter `QAsystem-BERT` and run
```
python QAsystemBERT.py
```
It will first display the number of question and answers pairs to train this model:
```
Number of question and answers: 108647
```
Then the model will ask you to enter a paragraph of text. You can use the following example:
```
The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, located in Vatican City. Formally established in 1475, although it is much older, it is one of the oldest libraries in the world and contains one of the most significant collections of historical texts. It has 75,000 codices from throughout history, as well as 1.1 million printed books, which include some 8,500 incunabula.   The Vatican Library is a research library for history, law, philosophy, science and theology. The Vatican Library is open to anyone who can document their qualifications and research needs. Photocopies for private study of pages from books published between 1801 and 1990 can be requested in person or by mail.   In March 2014, the Vatican Library began an initial four-year project of digitising its collection of manuscripts, to be made available online.   The Vatican Secret Archives were separated from the library at the beginning of the 17th century; they contain another 150,000 items.   Scholars have traditionally divided the history of the library into five periods, Pre-Lateran, Lateran, Avignon, Pre-Vatican and Vatican.   The Pre-Lateran period, comprising the initial days of the library, dated from the earliest days of the Church. Only a handful of volumes survive from this period, though some are very significant.
```
And then you can ask questions with the guide. There is a little demo:
```
Please enter your question: 
When was the Vat formally opened?

Predicted answer:
1475

Do you want to ask another question based on this text (Y/N)?Y

Please enter your question: 
What is the library for?

Predicted answer:
Research library for history , law , philosophy , science and theology

Do you want to ask another question based on this text (Y/N)?Y

Please enter your question: 
for what subjects?

Predicted answer:
History , law , philosophy , science and theology

Do you want to ask another question based on this text (Y/N)?N

Bye!
```
## Android app

The detailed progress information could be find in `plan`.
