To run the following code, `pip install pandas, numpy, pyspark`. The expected outputs are as follow:


# Part 1: Mapreduce
Top 5 words for document 0:
  cynics: 0.5637
  wall: 0.5116
  claw: 0.4991
  dwindling: 0.4572
  sellers: 0.4468

Top 5 words for document 1:
  carlyle: 0.7168
  occasionally: 0.3327
  timed: 0.3245
  bets: 0.2786
  aerospace: 0.2581

Top 5 words for document 2:
  outlook: 0.4265
  doldrums: 0.3770
  economy: 0.3721
  depth: 0.3134
  hang: 0.3048

Top 5 words for document 3:
  pipeline: 0.4721
  main: 0.3649
  oil: 0.3576
  southern: 0.3366
  flows: 0.2774

Top 5 words for document 4:
  menace: 0.5747
  tearaway: 0.3919
  straining: 0.2904
  toppling: 0.2796
  wallets: 0.2665
+---+-------------------+----------+
| id|              tfidf|      word|
+---+-------------------+----------+
|  0| 0.5115985326511431|      wall|
|  0| 0.2584728642725166|        st|
|  0| 0.3372044607529448|     bears|
|  0|  0.499114829314058|      claw|
|  0| 0.1892216338539946|      back|
|  0| 0.2953171727366614|     black|
|  0|0.24754017186645658|   reuters|
|  0| 0.2773120373951269|     short|
|  0| 0.4468379768438066|   sellers|
|  0|0.24678348986493034|    street|
|  0| 0.4572386180709258| dwindling|
|  0| 0.3643421454792778|      band|
|  0| 0.4125512394225831|     ultra|
|  0|  0.563734318747707|    cynics|
|  0|0.37743394553516213|    seeing|
|  0| 0.2877107940095433|     green|
|  1| 0.7168306746824437|   carlyle|
|  1| 0.1973537176743789|     looks|
|  1| 0.1898997183872362|    toward|
|  1| 0.2057832028092643|commercial|
+---+-------------------+----------+
only showing top 20 rows

# Part 2: SVM objective function
SVM Loss: 0.9998
[-1, -1, -1, 1, -1]