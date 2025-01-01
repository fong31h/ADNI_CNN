# This is the example of saving the vector embeddings


# as_features_list = []
# with torch.no_grad():
#     for batch in dataloader_2d_test:
#         test_matrix = batch['matrix'].to(device, dtype=torch.float32)
#         test_matrix = torch.unsqueeze(test_matrix, 1)
#         test_label = batch['label']
#         features = resnet50_dimred(test_matrix)
#         as_features_list.append(features)