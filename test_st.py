weights = "/data/quynhpt/tranformer_ocr_vn/vietocr-master/weights/transformerocr.pth"
# model = model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
pretrained_dict = torch.load(weights, map_location=torch.device(device))
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model.load_state_dict(pretrained_dict, strict=False)
