from torchvision import transforms

def create_transforms(aug_config):
    ''' Функция для создания трансформаций из конфига '''
    transform_list = []
    
    # Resize
    if 'resize' in aug_config:
        transform_list.append(transforms.Resize(tuple(aug_config['resize'])))
    
    # Аугментации
    if aug_config.get('random_horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())
        
    if 'random_rotation' in aug_config:
        transform_list.append(transforms.RandomRotation(aug_config['random_rotation']))
    
    if 'color_jitter' in aug_config:
        transform_list.append(transforms.ColorJitter(
            brightness=aug_config['color_jitter']['brightness'],
            contrast=aug_config['color_jitter']['contrast'],
            saturation=aug_config['color_jitter']['saturation'],
            hue=aug_config['color_jitter']['hue']
        ))
    
    if 'random_affine_translate' in aug_config:
        transform_list.append(transforms.RandomAffine(
            degrees=0,
            translate=tuple(aug_config['random_affine_translate'])
        ))
    
    if 'random_perspective' in aug_config:
        transform_list.append(transforms.RandomPerspective(
            distortion_scale=aug_config['random_perspective']['distortion_scale'],
            p=aug_config['random_perspective']['p']
        ))
    
    if 'random_crop' in aug_config:
        transform_list.append(transforms.RandomCrop(aug_config['random_crop']))
    
    # Обязательные преобразования
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['normalize']['mean'],
            std=aug_config['normalize']['std']
        )
    ])
    
    return transforms.Compose(transform_list)