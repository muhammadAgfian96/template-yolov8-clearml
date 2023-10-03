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

    def filter_category(self, exclude_class:List[str]=[])->List[str]:
        print("len_categories", len(self.categories))
        print("len_images", len(self.images))
        print("len_annotations", len(self.annotations))

        if len(exclude_class) == 0:
            return

        exclude_class  = [lbl.lower() for lbl in exclude_class]

        map_old_to_new_categories = {}
        new_category = []
        new_id = 0
        for cat in self.categories:
            if cat.name.lower() not in exclude_class: 
                map_old_to_new_categories[cat.id] = new_id
                new_category.append(Category(
                    id=new_id,
                    name=cat.name.lower(),
                    supercategory=cat.supercategory
                ))
                new_id += 1

        new_ls_annotations = []
        ls_imgs_id_without_ann = []

        for ann in self.annotations:
            new_cat_id = map_old_to_new_categories.get(ann.category_id)
            if new_cat_id:
                ann.category_id = new_cat_id
                new_ls_annotations.append(ann)
            else:
                ls_imgs_id_without_ann.append(ann.image_id)

        for idx, ann in enumerate(new_ls_annotations):
            ann.id = idx

        # remove images without annotations
        ls_new_images = []
        for img in self.images:
            if img.id not in ls_imgs_id_without_ann:
                ls_new_images.append(img)
        
        for idx, image in enumerate(ls_new_images):
            image.id = idx

        self.images = ls_new_images
        self.categories = new_category
        self.annotations = new_ls_annotations

        print("NEW len_categories", len(self.categories))
        print("NEW len_images", len(self.images))
        print("NEW len_annotations", len(self.annotations))

    def get_categoryid_to_namecat(self)->Dict[int, str]:
        categories_map = {}
        # exclude_class  = [lbl.lower() for lbl in exclude_class]
        # if len(exclude_class) > 0:
        #     print("WE EXCLUDE CLASS:", exclude_class)

        for cat in self.categories:
            categories_map[cat.id] = cat.name
        return categories_map

    def get_imageid_to_image(self) -> Dict[int, Image]:
        image_map = {}
        for img in self.images:
            image_map[img.id] = img
        return image_map
    
    def get_imageid_to_annotations(self, 
            exclude_class:List[str]=[],  
            attributes_excluded:Dict[str, str]=None,
            area_segment_min:float=None
        )->Dict[int, List[Annotation]]:
        """
        This function will return a dictionary of image_id to annotations
        and filters in level annotations happen here
        """
        if exclude_class is None:
            exclude_class = []

        imageid2anns = defaultdict(list)
        exclude_class  = [lbl.lower() for lbl in exclude_class]
        id2label = self.get_categoryid_to_namecat(exclude_class=exclude_class)
        print(id2label)
        

        for ann in self.annotations:
            skip = False
            if area_segment_min is not None:
                if ann.area < area_segment_min:
                    print(f"[Area Filters Active] minimal: {area_segment_min} | actual:", ann.area)
                    skip = True
                    continue
                
            if attributes_excluded is not None:
                for attr_name, attr_value in attributes_excluded.items():
                    attributes_dataset = set(ann.attributes.get(attr_name).replace(", ", ",").replace(" ,", ",").split(","))
                    attributes_config = set(attr_value.replace(", ", ",").replace(" ,", ",").split(","))
                    if attributes_dataset is None:
                        continue
                    
                    # print("attributes_excluded", attributes_excluded, ann.attributes)
                    if isinstance(attributes_dataset, set):
                        intersection = attributes_config.intersection(attributes_dataset)
                        if intersection:
                            print("[Attributes Filters Active]", attr_name, attributes_config, attributes_dataset)
                            skip = True
                            break
                        break

            if id2label[ann.category_id] in exclude_class:
                # print("[Class Filters]",id2label[ann.category_id], "Class Exclude")
                skip = True
            
            if skip:
                continue

            imageid2anns[ann.image_id].append(ann)

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

