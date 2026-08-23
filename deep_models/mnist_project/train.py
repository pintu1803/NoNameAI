##Train configs are stored in config.py

from config import TrainConfig

def train_model(modelObj, train_loader, val_loader):
    epochs = TrainConfig.epoch_count
    checkpoint = TrainConfig.epoch_checkpoint

    modelObj.showModel()

    loss_fn = modelObj.loss_fn()

    optimizer = modelObj.optimizer()

    model = modelObj.deepModel

    #TRAIN and Validate
    for epoch in range(epochs):

        #Training mode ON
        model.train()

        train_loss = 0
        for image, label in train_loader:

            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #add loss for all batches
            train_loss += loss

        #Validation mode ON
        model.eval()

        val_loss = 0
        for image, label in val_loader:
            pred = modelObj.forward(image)
            loss = loss_fn(pred, label)  
            val_loss += loss   

        #print for every checkpoint-th epoch
        if epoch % checkpoint == 0:
            print(f"\nLoss computed - epoch : {epoch}")
            print(f"Train Loss : {train_loss : .3f}, Validation Loss : {val_loss : .3f}")

