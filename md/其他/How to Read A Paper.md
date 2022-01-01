[TOC]





# Three-pass Approach

The three-pass approach is acted as a filtering system. It's an iterative and incremental way of reading a paper. 

- general overview to specific details
- give you deeper insights in each iteration



## 1. The first pass: The bird's-eye view

bird's-eye view or the big picture of the paper（5-10分钟） 

- skim through the structure of the paper
  - look at the sections and subsections but ignore their content
  - come up with vague questions
- ignore any details like math equations
- you should read
  - **abstract**
  - **title**
  - **introduction** 
  - **conclusion**

- check if the paper is worth reading in general
- discard papers which are not helpful

after the first pass, you should be able to answer the so-called "five C's"

1. Category: the type of the paper
2. Context: the relations with other papers
3. Correctness: validity measurement；assumption valid?
4. Contributions: Are these contributions meaningful?
5. Clarity: is the paper well written? grammar mistake? typos?

**DO NOT CONTINUE READING IF...**

- lack background information
- do not know enough about the topic
- not interest you
- poorly written
- false assumptions



## 2. The second pass: Grasp the content

understand the content of the paper by reading it as a whole（1 hour）

- ignore details like math equations or proofs
- make notes at the margins and write down key points
- rephrase the key points in your own words(summaries)
- look at illustration like tables and figures and see if you can spot any mistake or discrepancies
  - do the illustrations make sense
  - the information they convey
  - are axes properly labeled?
  - proper captions?
- mark relevant unread references for further reading to learn more about the background



## 3. The third pass: virtually re-implement the paper

be certain that if this paper is worth your time（5 hours）

- read the complete paper with all its math equations and details
  - make the same assumptions as the authors
  - re-create the work from scratch
- virtually re-implement the paper in your head
- use any tools to recreate the results
  - draw a flowcharts of the different steps 
  - use pseudo-code
  - re-implement things in python

**AFTER THE THIRD PASS YOU CAN**

know the paper's strong and weak points. 

make statements about missing citations and potential issues

reconstruct the structure and explain to someone in simple language what the paper is all about



# Doing a Literature Survey



## 1. First Pass

- collect potentially usefull papers
  - use google scholar to search keywords and find 3-5 **recent** papers
  - create a simple list of paper clusterd by their topic、year、citation
  - ![20201124173647201.png (879×356) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124173647201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)
- after have your little collection of initial papers ready, continue with the usual first-pass on each of them to get the big picture.
- skim through the references to see if the papers have any citation in common

## 2. Second Pass

- when you identified common citations and repeated authors, you can visit their websites and see if you can spot any recent work

- download the commonly cited papers and apply the three-pass approach for single papers

## 3. Third Pass

- visit the website of the top conferences or journals and look through the recent proceedings.

- identify related hight-quality work and apply the three-pass approach for single papers





# Optional Extensions

if you already know you have to read and understand the paper entirely, these steps might help.

**Little Boxes**

![20201124182953839.png (989×556) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124182953839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

- surrounded math equations, figures and tables with boxes. 
- clearly separated boxes and separate the text from the rest. 
- do this during the first-pass 
- this helps to quantify how many details in terms of math equations

**Highlighters**

- using highlights to mark section in your paper and give them distinctive meanings
- come up with your own highlighting system or using existing one, during second-pass:
  - use **yellow** for interesting or important sentences.
  - **Orange** is for citations
  - **Green** is for definitions or catchphrases
- During the second pass , you should take notes at the margins, draw little diagrams for better understanding and use highlighters in combination

![202011241840360.png (989×556) (csdnimg.cn)](https://img-blog.csdnimg.cn/202011241840360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)



Interesting or important references at the end of the paper get the same color as before.

![20201124184238296.png (989×556) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124184238296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

**Mindmaps**

- start with the title of the paper in the center
- big arrows are pointing to the main section titles & highlighted with orange
- first-level subsection are highlighted with green
- anything else get no highlighting

![20201124184321950.png (989×556) (csdnimg.cn)](https://img-blog.csdnimg.cn/20201124184321950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

**Pomodoro sessions**

- a great tool if you are lacking motivation
- get a timer and set it to 25 minutes
- eliminate any distractions and follow the three-pass approach until the 25 minutes are up

**Feynman technique and rubber duck debugging**

Feynman:  general steps are:

- **choose a concept** you want to learn and write its name at the top of a piece of paper
- **pretend you are teaching the concept** to someone who has no prior knowledge about it. try to use simplest language and do not simply recite. Use your own words!
- **review your explanation** accurate ? identify weak points in your explanations and write it on a piece of paper.
- **Simplify your explanation** do not use technical terms or complex language 

Rubber duck debugging:

- a programmer carries around a rubber duck and explain the code, line by line 
- and spot any mistakes

![2020112418433011.png (989×556) (csdnimg.cn)](https://img-blog.csdnimg.cn/2020112418433011.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70#pic_center)

**Parkinson’s law and the Pareto principle**

[**Parkinson’s law**](https://en.wikipedia.org/wiki/Parkinson's_law) states the following:

> “Work expands so as to fill the time available for its completion” — **Cyril Northcote Parkinson**

If you plan 10 hours to read a paper, taking notes, writing summaries and so forth, then it will probably take you 10 hours.

**pareto principle(80/20)**

> “For many events, roughly 80% of the effects come from 20% of the causes.” — Vilfredo Pareto

This means that it takes you probably 20% of your overall effort and time to understand 80% of the paper. This 80/20 split is not fixed but is rather a rough estimate. 





## How to Read Scientific Papers(from a video)

1. Understand the title
2. Don't pay much attention to authors
3. Read abstract well and form hypothesis
4. Look at pictures to understand solutions
5. Read Introduction Carefully
6. Skip related work
7. Read everything else and skip things that don't seem like they are part of the general idea
8. Look at the results and try to prove them wrong or get convinced that solution works
9. Glance over everything and make sure all your questions were answered
10. Read it a couple of times or take breaks to have a good understanding

