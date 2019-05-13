**Project: Design a Recommendation System**

Recommendation Systems are an increasingly popular application of Machine Learning in many industries. If you've used services like 
Amazon, Netflix or Pandora, you might notice their personalized recommendations suggesting items for you to buy, movies to watch, 
or songs to listen to. An example of how influential a well-designed recommendation system can be is what happened to Joe Simpson's 
Touching the Void, a book about mountain-climbing. The book was not a bestseller when it was published originally in 1988, but 
surged in popularity after another mountain-climbing book, Into Thin Air by Jon Krakauer, topped the charts in 1997. 
This happened because Amazon's recommendation system noticed that a few people who bought and enjoyed Into Thin Air also bought 
and enjoyed Touching the Void. Without good online recommendation systems to suggest the book to readers, 
Touching the Void may have been quickly forgotten. Instead, it became very popular in its own right and in some respects 
exceeded the popularity of Into Thin Air.

Similarly, in this project, we will recommend movies to users. We will use the MovieLens dataset 
(https://www.udacity.com/api/nodes/1021838736/supplemental_media/ml2-projectzip/download  or https://grouplens.org/datasets/movielens/)
to create a recommendation engine like Netflix's. 
In a recommendation-system application, there are two classes of entities, users and items. 
Users have preferences for certain items and these preferences must be discovered from the data. 
The data is represented as a utility matrix, a value that represents the rating given by that user 
for that item and is given for each user-item pair. The ratings can take a value between 1 to 5, 
representing the number of stars that the user gave as a rating for that item.
The matrix is sparse, meaning that most of the entries are unknown. The goal of the recommendation 
engine is to predict the blanks in a utility matrix. 	After filling the blanks, for a given user
we recommend the item, in our case film,  which  is not rated, not watched, by that user and has the 
highest value in our predictions related to that user. 

In part 1  we will use Collaborative Filtering to build our recommendation engine. Collaborative Filtering 
Systems recommend items based on similarity measures between users and/or items. The items they recommend 
to a user are drawn from those preferred by similar users. For example, if an Amazon shopper places a toy robot in their cart, 
then a robot-themed Lego set is recommended because other shoppers who liked the toy robot also liked the Lego set. 
In order to make these predictions, we  first measure similarity of users or items from the rows and columns of the Utility Matrix. 
We use the Pearson Correlation formula. But unfortunately the utility matrix is too sparse to draw any reliable conclusions.
We will see that this approach brings no improvement over results provided by almost trivial approaches. 

In part 2 we will predict ratings via a machine learning algorithm called random forest. In this case our training data 
is ratings dataset. Using datasets on films and users we extend ratings  data horizontally. After the tuning of parameters
we improve the results coming from first part substantially. 

Jupyter notebook files can be read in https://nbviewer.jupyter.org if there is any need. 
