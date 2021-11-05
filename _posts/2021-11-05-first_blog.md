# C1&C2 of fastai lesson

this is my first blog recording my learning procedure of ML, inspried by [Rachel Thomas's blog](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045).

Hope it will lasts for as long as I could.

And if there's really anyone reading this, thanks and enjoy!

since I started this blog in the C2, and I don't have much time to review the previous chapter, I'll just start at the end of Chapter 2, maybe I will add some notes of Chapter 1 sometime...
## C2-production
### Questionair

1.Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.

	Sometimes the data we used for trainning are not similar of the real data in production, like the pictures in the night , and maybe the production don't even offer us pictures but video clips to be as the input.
 
2.Where do text models currently have a major deficiency?
	 
	 I'm not sure, is it hard to process the text into vector, causing curse of dimentionality ?
	 
	 Or, because for text generative model, there is no label to identify the text generated is true or false.
	 
	 Take an example ,deep learning is not good at generating *correct* responses, so when it comes to a highly sophisticated area like medical industry ,a subtle mistake will damage life security.
	 
	 And in social media , information that seems true but artificial may cause conflicts and damages, so this tech is some kind of double-edged-sword.
3.In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
	 
	 Maybe don't use the model in production immediately, but as an assistant for manual work, when the model got more information in the real production , and it did well in accuracy, it's time to consider make it  rolling out independently.
	 
4.What kind of tabular data is deep learning particularly good at?
 
	 Deep learning does greatly increase the variety of columns that you can include—for example, columns containing natural language (book titles, reviews, etc.), and high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID).
	 
5.What's a key downside of directly using a deep learning model for recommendation systems?
 
       However, nearly all machine learning approaches have the downside that they only tell you what products a particular user might like, rather than what recommendations would be helpful for a user. 
       
6.What are the steps of the Drivetrain Approach?

	find a *objective*, spot what *levers* we have ,gather the *data* we have or need , and then build a *model* that you can use to determine the best actions to take to get the best results in terms of your objective.

7.How do the steps of the Drivetrain Approach map to a recommendation system?

	Let's consider another example: recommendation systems. The *objective* of a recommendation engine is to drive additional sales by surprising and delighting the customer with recommendations of items they would not have purchased without the recommendation. The *lever* is the ranking of the recommendations. New *data* must be collected to generate recommendations that will *cause new sales*. This will require conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers. This is a step that few organizations take; but without it, you don't have the information you need to actually optimize recommendations based on your true objective (more sales!).
	Finally, you could build two *models* for purchase probabilities, conditional on seeing or not seeing a recommendation. The difference between these two probabilities is a utility function for a given recommendation to a customer. It will be low in cases where the algorithm recommends a familiar book that the customer has already rejected (both components are small) or a book that they would have bought even without the recommendation (both components are large and cancel each other out).
8.Create an image recognition model using data you curate, and deploy it on the web.

	I do create a low accuracy age detection model using the data teacher gave us, but I didn't deploy it on the web, maybe when I did some really great work, that I will reread this section and deploy it.
	
9.What is  `DataLoaders`?

	`DataLoaders` is a thin class that just stores whatever `DataLoader` objects you pass to it, and makes them available as `train` and `valid`. 
	
10.What four things do we need to tell fastai to create  `DataLoaders`?

	- What kinds of data we are working with
	- How to get the list of items
	- How to label these items
	- How to create the validation set
	
11.What does the  `splitter`  parameter to  `DataBlock`  do?

	It split the data into training set and validation set.

12.How do we ensure a random split always gives the same validation set?

	Use the same random seed.
	
13.What letters are often used to signify the independent and dependent variables?

	`x`: independent variable
	`y`: dependent variable
	
14.What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?

	`crop`  the images to fit a square shape of the size requested, using the full width or height. This can result in losing some important details. Alternatively, you can ask fastai to pad the images with zeros (black), or squish/stretch them;

15.What is data augmentation? Why is it needed?

	`Data augmentation` refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes and contrast changes.

16.What is the difference between  `item_tfms`  and  `batch_tfms`?

	 item_tfms put the transforms on a single image,while batch_tsfm is on a batch size if images, because the transform function remains the same for every image, so we will have a speed promotion when using batch_tsfm on GPU.

17.What is a confusion matrix?

	something throught which you can have a insight of how many category is correctly classified and incorrectly classified.

18.What does  `export`  save?

	the *architecture* and the trained *parameters*

19.What is it called when we use a model for getting predictions, instead of training?

	When we use a model for getting predictions, instead of training, we call it *inference*. 

20.What are IPython widgets?

	*IPython widgets* are GUI components that bring together JavaScript and Python functionality in a web browser, and can be created and used within a Jupyter notebook.

21.When might you want to use CPU for deployment? When might GPU be better?

	As we've seen, GPUs are only useful when they do lots of identical work in parallel. If you're doing (say) image classification, then you'll normally be classifying just one user's image at a time, and there isn't normally enough work to do in a single image to keep a GPU busy for long enough for it to be very efficient. So, a CPU will often be more cost-effective.
	An alternative could be to wait for a few users to submit their images, and then batch them up and process them all at once on a GPU. But then you're asking your users to wait, rather than getting answers straight away!

22.What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
	
	There are downsides too, of course. Your application will require a network connection, and there will be some latency each time the model is called. (It takes a while for a neural network model to run anyway, so this additional network latency may not make a big difference to your users in practice. In fact, since you can use better hardware on the server, the overall latency may even be less than if it were running locally!) Also, if your application uses sensitive data then your users may be concerned about an approach which sends that data to a remote server, so sometimes privacy considerations will mean that you need to run the model on the edge device (it may be possible to avoid this by having an *on-premise* server, such as inside a company's firewall). Managing the complexity and scaling the server can create additional overhead too, whereas if your model runs on the edge devices then each user is bringing their own compute resources, which leads to easier scaling with an increasing number of users (also known as *horizontal scaling*)

23.What are three examples of problems that could occur when rolling out a bear warning system in practice?What is "out-of-domain data"?What is "domain shift"?

	out-of-domain data:
	Working with video data instead of images
	Handling nighttime images, which may not appear in this dataset
	Dealing with low-resolution camera images
	Ensuring results are returned fast enough to be useful in practice
	Recognizing bears in positions that are rarely seen in photos that people post online (for example from behind, partially covered by bushes, or when a long way away from the camera)
	
	domain shift:
	as time pass by,the original training data is no longer  representive under the latest situation


24.What are the three steps in the deployment process?

	first step is to use an entirely manual process, with your deep learning model approach running in parallel but not being used directly to drive any actions.
	The second step is to try to limit the scope of the model, and have it carefully supervised by people.
	Then, gradually increase the scope of your rollout. 


	
