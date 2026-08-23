##Contains absolute method, no method

def test_model(modelObj, test_loader):

    modelObj.deepModel.eval()
    loss_fn = modelObj.loss_fn()
    #TEST the model
    print("\nTESTING TIME\n")
    for idx, (image, label) in enumerate(test_loader):
        pred = modelObj.forward(image)
        loss = loss_fn(pred, label)  

        #print for every 10th batch
        if idx % 10 == 0:
            print(f"\nLoss computed - epoch : {idx}")
            print(f"Testing Loss : {loss : .3f}")