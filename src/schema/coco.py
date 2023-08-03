from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import defaultdict

class License(BaseModel):
    id: int
    name: str
    url: str

class Info(BaseModel):
    year: str
    version: str
    description: str
    contributor: str
    url: str
    date_created: str

class Category(BaseModel):
    id: int
    name: str
    supercategory: Optional[str]

class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flickr_url: Optional[str]
    coco_url: Optional[str]
    date_captured: int

class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]]
    area: float
    bbox: List[float]
    iscrowd: int
    attributes: dict


class Coco(BaseModel):
    licenses: List[License]
    info: Info
    categories: List[Category]
    images: List[Image]
    annotations: Optional[List[Annotation]]

    def get_categoryid_to_namecat(self, exclude_class:List[str]=[])->Dict[int, str]:
        categories_map = {}
        exclude_class  = [lbl.lower() for lbl in exclude_class]
        print("WE EXCLUDE CLASS:", exclude_class)
        for cat in self.categories:
            if cat.name not in exclude_class:
                categories_map[cat.id] = cat.name
        return categories_map

    def get_imageid_to_image(self) -> Dict[int, Image]:
        image_map = {}
        for img in self.images:
            image_map[img.id] = img
        return image_map
    
    def get_imageid_to_annotations(self, exclude_class:List[str]=[])->Dict[int, List[Annotation]]:
        imageid2anns = defaultdict(list)
        exclude_class  = [lbl.lower() for lbl in exclude_class]
        id2label = self.get_categoryid_to_namecat()
        print(id2label)

        for ann in self.annotations:
            if id2label[ann.category_id] not in exclude_class:
                imageid2anns[ann.image_id].append(ann)
            else:
                print(id2label[ann.category_id], "Exclude")
        return imageid2anns


    def checking_task(self):
        task_type = set()
        if len(self.annotations) == 0:
            print("No annotations")
            return list(task_type)

        for ann in self.annotations:
            if len(ann.bbox) != 0:
                task_type.add("detection")
            if len(ann.segmentation) != 0:
                task_type.add("segmentation")
        return list(task_type)            

