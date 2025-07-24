python -c "
import os
def count_images(path):
    if os.path.exists(path):
        return len([f for f in os.listdir(path) if f.endswith('.jpg')])
    return 0

print('📊 Dataset Verification:')
total = 0
for split in ['train', 'val', 'test']:
    print(f'\n{split.upper()} SET:')
    for class_name in ['real', 'fake']:
        count = count_images(f'data/processed/{split}/{class_name}')
        print(f'  {class_name}: {count:,} images')
        total += count
print(f'\n🎯 TOTAL IMAGES: {total:,}')
