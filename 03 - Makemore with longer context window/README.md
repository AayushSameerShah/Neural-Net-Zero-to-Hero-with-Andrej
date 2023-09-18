# 🌳 Making the mature model

We will increate the context from just past **single token** to past **three tokens**. And that requires us to introduce a couple of other components to be considered: the shapes.

## 📔 In this section:

📂 We will need to create a proper dataset which can effectively be used to provide past **three tokens** to predict the next.<br>
🧩 We tune the embeddings size. Here you will have your first time interaction with the embeddings for the tokens!<br>
🧐 We will compare how the embeddings change over the time, **the common characters** will come closer and go further if they represent the different sementics! *(before and after the training)*<br>

> 😉 It introduces **an amazing realization** in between the discussion, we will finally unveil the meaning of jargon **binary cross entropy loss**! Which we already knew but just didn't name it!
