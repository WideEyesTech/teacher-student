    
# def training():

#     dataloader = torchvision.dataset.DatasetLoader(dataset)

#     optimzer = SGD.optimizer()

#     # Adam

#     model = model

#     model.cuda()

#     acc_grad = 10

#     for epoch in range(100):
#         model.train()
#         model.zero_grad()
#         for idx, batch in enumerate(dataloader):
#             X = batch['images']
#             y = batch['labels']

#             X = X.to('cuda')
#             pred = model(X)


#             loss = compute_loss(y, pred)

#             loss /= acc_grad

#             loss.backward()

#             if idx % acc_grad == 0:
#                 optimizer.step()
#                 model.zero_grad()

#         model.eval()
#         metric = do_evaluation(dataloader_val, model)
#         print('epoch {} perf: {}'.format(epoch, metric))

#         torch.save(model)