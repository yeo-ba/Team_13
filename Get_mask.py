from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


model_name = "facebook/sam-vit-huge"

class Get_mask:
    def __init__(self, device, img_path, model_name = model_name):
        self.img_path = img_path
        self.device = device
        self.raw_image = Image.open(img_path).convert("RGB")
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
    
    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

    def show_boxes_on_image(self, raw_image, boxes):
        plt.figure(figsize=(10,10))
        plt.imshow(raw_image)
        for box in boxes:
            self.how_box(box, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_on_image(self, input_points, input_labels=None):
        plt.figure(figsize=(10,10))
        plt.imshow(self.raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        plt.axis('on')
        plt.show()

    def show_points_and_boxes_on_image(self, boxes, input_points, input_labels=None):
        plt.figure(figsize=(10,10))
        plt.imshow(self.raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis('on')
        plt.show()


    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_masks_on_image(self, masks, scores):
        if len(masks.shape) == 4:                           # 텐서가 4차원이면 
            masks = masks.squeeze()                           # 1인 값을 제거 
        if scores.shape[0] == 1:                            # 배치 사이즈가 1이면 
            scores = scores.squeeze()                         # 1인 값을 제거 

        nb_predictions = scores.shape[-1]
        fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask = mask.cpu().detach()
            axes[i].imshow(np.array(self.raw_image))
            self.show_mask(mask, axes[i])
            axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
            axes[i].axis("off")
        plt.show()
    
    def get_mask(self, input_points):
        inputs = self.processor(self.raw_image, input_points=input_points, return_tensors="pt").to(self.device)
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores

        max_score_index = torch.argmax(scores[0][0])  # 스코어 값 중 가장 큰 값의 인덱스
        max_score = torch.max(scores[0][0]).item()
        max_score_mask = masks[0][:, max_score_index, :, :]     # 가장 스코어가 큰 마스크만 가져오기 
        self.max_score_mask = max_score_mask
        color = np.array([0,0,0,1.0])                        # R, G, B, 투명도(0:완전 투명, 1은 불투명) 값

        h, w = max_score_mask.shape[-2:]
        mask_image = max_score_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image_np = mask_image.numpy() 
        
        raw_image_tensor = transforms.ToTensor()(self.raw_image)                      # 이미지를 pt 텐서로 변환하는 함수            

        masked_image = raw_image_tensor * max_score_mask           # 불리안 마스크 곱해서 -> True 영역만 살리기

        self.masked_image = transforms.ToPILImage()(masked_image)
        plt.axis('off')
        plt.imshow(mask_image_np)
        plt.savefig('bg_image.png', bbox_inches='tight', pad_inches=0)

    def get_masked_image(self):

        return self.masked_image