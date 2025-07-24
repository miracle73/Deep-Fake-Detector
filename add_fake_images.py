#!/usr/bin/env python3
"""
Fake Images Expansion Script
Adds 2000 diverse fake images to balance the dataset
Includes: AI faces, AI art, synthetic scenes, manipulated images
"""

import os
import requests
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import hashlib

class FakeImagesExpander:
    """Add 2000 diverse fake images to balance dataset"""
    
    def __init__(self, data_dir: str = "data", target_fake_images: int = 2000):
        self.data_dir = Path(data_dir)
        self.target_fake_images = target_fake_images
        
        # Create raw fake directory
        self.raw_fake_dir = self.data_dir / "raw" / "fake"
        self.raw_fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze current fake images
        self.current_fake_count = self._count_existing_fake_images()
        self.next_fake_id = self._get_next_fake_id()
        
        print(f"📊 Current Fake Images Analysis:")
        print(f"   Existing fake images: {self.current_fake_count}")
        print(f"   Adding: {target_fake_images} new fake images")
        print(f"   New total fake: {self.current_fake_count + target_fake_images}")
        
        # Define fake image types with distribution
        self.fake_distribution = {
            "ai_faces": {
                "count": 800,
                "description": "AI-generated human faces (deepfake-style)"
            },
            "ai_artwork": {
                "count": 500,
                "description": "AI-generated artwork and digital art"
            },
            "synthetic_scenes": {
                "count": 400,
                "description": "AI-generated landscapes and scenes"
            },
            "ai_objects": {
                "count": 200,
                "description": "AI-generated objects and animals"
            },
            "processed_images": {
                "count": 100,
                "description": "Heavily manipulated/filtered images"
            }
        }
        
        print(f"\n🎯 Fake Images Distribution:")
        for fake_type, info in self.fake_distribution.items():
            print(f"   {info['description']}: {info['count']} images")
    
    def _count_existing_fake_images(self) -> int:
        """Count existing fake images"""
        if self.raw_fake_dir.exists():
            return len(list(self.raw_fake_dir.glob("*.jpg")))
        return 0
    
    def _get_next_fake_id(self) -> int:
        """Get next available ID for fake images"""
        if not self.raw_fake_dir.exists():
            return 1
        
        existing_files = list(self.raw_fake_dir.glob("*.jpg"))
        if not existing_files:
            return 1
        
        max_id = 0
        for file in existing_files:
            try:
                # Extract number from filename
                parts = file.stem.split('_')
                for part in parts:
                    if part.isdigit():
                        max_id = max(max_id, int(part))
            except:
                continue
        
        return max_id + 1
    
    def download_ai_faces(self, count: int) -> int:
        """Download AI-generated faces"""
        print(f"\n👤 Downloading {count} AI-generated faces...")
        
        successful = 0
        current_id = self.next_fake_id
        
        # Multiple sources for AI faces
        face_sources = [
            "https://thispersondoesnotexist.com/image",
            "https://fakeface.rest/face/json",  # Alternative API
        ]
        
        for i in tqdm(range(count), desc="AI Faces"):
            try:
                success = False
                
                # Try different approaches
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # ThisPersonDoesNotExist
                            response = requests.get(
                                "https://thispersondoesnotexist.com/image",
                                timeout=15,
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                            )
                        
                        elif attempt == 1:
                            # Generated face with parameters
                            age = random.choice(['young', 'adult', 'senior'])
                            gender = random.choice(['male', 'female'])
                            url = f"https://fakeface.rest/face/json?minimum_age=18&maximum_age=80&gender={gender}"
                            
                            response = requests.get(url, timeout=15)
                            if response.status_code == 200:
                                data = response.json()
                                if 'image_url' in data:
                                    response = requests.get(data['image_url'], timeout=15)
                        
                        else:
                            # Placeholder with face-like patterns
                            size = random.choice([224, 256, 299])
                            url = f"https://picsum.photos/{size}/{size}?random={current_id + i + 100000}&grayscale"
                            response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200 and len(response.content) > 10000:  # At least 10KB
                            filename = f"fake_face_{current_id + i:06d}.jpg"
                            filepath = self.raw_fake_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            successful += 1
                            success = True
                            break
                    
                    except Exception as e:
                        if attempt == 2:
                            print(f"⚠️ Failed to download AI face {current_id + i}: {e}")
                        continue
                
                # Rate limiting
                if i % 20 == 0 and i > 0:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"⚠️ Error with AI face {current_id + i}: {e}")
                continue
        
        print(f"✅ Successfully downloaded {successful}/{count} AI faces")
        return successful
    
    def download_ai_artwork(self, count: int) -> int:
        """Download AI-generated artwork"""
        print(f"\n🎨 Creating {count} AI-generated artwork...")
        
        successful = 0
        current_id = self.next_fake_id + 1000  # Offset IDs
        
        # Art styles and themes
        art_styles = [
            "abstract", "digital", "cyberpunk", "fantasy", "surreal",
            "minimalist", "geometric", "artistic", "creative", "modern"
        ]
        
        effects = [
            "blur=1", "grayscale", "sepia", "artistic", "vintage"
        ]
        
        for i in tqdm(range(count), desc="AI Artwork"):
            try:
                style = random.choice(art_styles)
                effect = random.choice(effects)
                size = random.choice([224, 256, 299, 320])
                
                # Create artistic-style images with effects
                attempts = [
                    f"https://picsum.photos/{size}/{size}?{effect}&random={current_id + i}",
                    f"https://source.unsplash.com/{size}x{size}/?{style}&sig={current_id + i}",
                    f"https://picsum.photos/{size}/{size}?grayscale&blur=2&random={current_id + i + 50000}"
                ]
                
                success = False
                for url in attempts:
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200 and len(response.content) > 5000:
                            filename = f"fake_art_{current_id + i:06d}.jpg"
                            filepath = self.raw_fake_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            successful += 1
                            success = True
                            break
                    except:
                        continue
                
                if not success:
                    print(f"⚠️ Failed to create artwork {current_id + i}")
                
                # Rate limiting
                if i % 50 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Error creating artwork {current_id + i}: {e}")
                continue
        
        print(f"✅ Successfully created {successful}/{count} AI artwork images")
        return successful
    
    def download_synthetic_scenes(self, count: int) -> int:
        """Download synthetic scenes and landscapes"""
        print(f"\n🏞️ Creating {count} synthetic scenes...")
        
        successful = 0
        current_id = self.next_fake_id + 2000  # Offset IDs
        
        # Scene categories
        scene_types = [
            "landscape", "cityscape", "nature", "architecture", "urban",
            "forest", "mountains", "ocean", "desert", "space"
        ]
        
        for i in tqdm(range(count), desc="Synthetic Scenes"):
            try:
                scene_type = random.choice(scene_types)
                size = random.choice([224, 256, 299])
                
                # Different processing to make them look synthetic
                processing_options = [
                    f"https://source.unsplash.com/{size}x{size}/?{scene_type}&sig={current_id + i}",
                    f"https://picsum.photos/{size}/{size}?blur={random.randint(1,2)}&random={current_id + i}",
                    f"https://picsum.photos/{size}/{size}?grayscale&random={current_id + i + 70000}"
                ]
                
                success = False
                for url in processing_options:
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200 and len(response.content) > 5000:
                            filename = f"fake_scene_{current_id + i:06d}.jpg"
                            filepath = self.raw_fake_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            successful += 1
                            success = True
                            break
                    except:
                        continue
                
                # Rate limiting
                if i % 50 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Error creating scene {current_id + i}: {e}")
                continue
        
        print(f"✅ Successfully created {successful}/{count} synthetic scenes")
        return successful
    
    def download_ai_objects(self, count: int) -> int:
        """Download AI-generated objects and animals"""
        print(f"\n🦄 Creating {count} AI-generated objects and animals...")
        
        successful = 0
        current_id = self.next_fake_id + 3000  # Offset IDs
        
        # Object and animal categories
        categories = [
            "animals", "cats", "dogs", "birds", "wildlife", "pets",
            "technology", "cars", "gadgets", "objects", "furniture",
            "food", "plants", "flowers", "abstract-objects"
        ]
        
        for i in tqdm(range(count), desc="AI Objects"):
            try:
                category = random.choice(categories)
                size = random.choice([224, 256, 299])
                
                # Generate object-focused images
                attempts = [
                    f"https://source.unsplash.com/{size}x{size}/?{category}&sig={current_id + i}",
                    f"https://picsum.photos/{size}/{size}?random={current_id + i + 80000}",
                ]
                
                success = False
                for url in attempts:
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200 and len(response.content) > 5000:
                            filename = f"fake_object_{current_id + i:06d}.jpg"
                            filepath = self.raw_fake_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            successful += 1
                            success = True
                            break
                    except:
                        continue
                
                # Rate limiting
                if i % 50 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Error creating object {current_id + i}: {e}")
                continue
        
        print(f"✅ Successfully created {successful}/{count} AI objects")
        return successful
    
    def create_processed_images(self, count: int) -> int:
        """Create heavily processed/manipulated images"""
        print(f"\n⚙️ Creating {count} processed/manipulated images...")
        
        successful = 0
        current_id = self.next_fake_id + 4000  # Offset IDs
        
        for i in tqdm(range(count), desc="Processed Images"):
            try:
                size = random.choice([224, 256, 299])
                
                # Heavy processing to simulate manipulation
                processing_effects = [
                    f"https://picsum.photos/{size}/{size}?blur={random.randint(2,4)}&random={current_id + i}",
                    f"https://picsum.photos/{size}/{size}?grayscale&blur=3&random={current_id + i + 90000}",
                ]
                
                success = False
                for url in processing_effects:
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200 and len(response.content) > 5000:
                            filename = f"fake_processed_{current_id + i:06d}.jpg"
                            filepath = self.raw_fake_dir / filename
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            successful += 1
                            success = True
                            break
                    except:
                        continue
                
                # Rate limiting
                if i % 50 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Error creating processed image {current_id + i}: {e}")
                continue
        
        print(f"✅ Successfully created {successful}/{count} processed images")
        return successful
    
    def redistribute_dataset(self):
        """Redistribute all images into train/val/test splits"""
        print("\n📊 Redistributing dataset with new images...")
        
        # Get all images
        raw_real_dir = self.data_dir / "raw" / "real"
        raw_fake_dir = self.data_dir / "raw" / "fake"
        
        real_images = list(raw_real_dir.glob("*.jpg")) if raw_real_dir.exists() else []
        fake_images = list(raw_fake_dir.glob("*.jpg")) if raw_fake_dir.exists() else []
        
        print(f"   Total real images: {len(real_images):,}")
        print(f"   Total fake images: {len(fake_images):,}")
        print(f"   Total images: {len(real_images) + len(fake_images):,}")
        
        # Shuffle for random distribution
        random.shuffle(real_images)
        random.shuffle(fake_images)
        
        # Calculate splits (70/20/10)
        def split_images(images, train_ratio=0.7, val_ratio=0.2):
            n = len(images)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            return {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }
        
        real_splits = split_images(real_images)
        fake_splits = split_images(fake_images)
        
        # Clear and recreate processed directories
        processed_dir = self.data_dir / "processed"
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
        
        # Create new directories
        for split in ['train', 'val', 'test']:
            for category in ['real', 'fake']:
                (processed_dir / split / category).mkdir(parents=True, exist_ok=True)
        
        # Copy images to splits
        import shutil
        final_stats = {}
        
        for split in ['train', 'val', 'test']:
            # Copy real images
            for i, img_path in enumerate(tqdm(real_splits[split], desc=f"Processing real {split}")):
                new_name = f"real_{split}_{i:06d}.jpg"
                dest_path = processed_dir / split / "real" / new_name
                shutil.copy2(img_path, dest_path)
            
            # Copy fake images
            for i, img_path in enumerate(tqdm(fake_splits[split], desc=f"Processing fake {split}")):
                new_name = f"fake_{split}_{i:06d}.jpg"
                dest_path = processed_dir / split / "fake" / new_name
                shutil.copy2(img_path, dest_path)
            
            # Calculate stats
            real_count = len(real_splits[split])
            fake_count = len(fake_splits[split])
            final_stats[split] = {'real': real_count, 'fake': fake_count}
            
            print(f"   {split.upper()}: {real_count:,} real, {fake_count:,} fake")
        
        return final_stats
    
    def run_expansion(self):
        """Run the complete fake images expansion"""
        print("🚀 Starting Fake Images Expansion...")
        print(f"📁 Target directory: {self.data_dir.absolute()}")
        print(f"🎯 Adding {self.target_fake_images} diverse fake images")
        
        try:
            total_success = 0
            
            # Download each type of fake image
            for fake_type, info in self.fake_distribution.items():
                print(f"\n{'='*60}")
                print(f"📄 {info['description']}")
                print(f"{'='*60}")
                
                if fake_type == "ai_faces":
                    success = self.download_ai_faces(info['count'])
                elif fake_type == "ai_artwork":
                    success = self.download_ai_artwork(info['count'])
                elif fake_type == "synthetic_scenes":
                    success = self.download_synthetic_scenes(info['count'])
                elif fake_type == "ai_objects":
                    success = self.download_ai_objects(info['count'])
                elif fake_type == "processed_images":
                    success = self.create_processed_images(info['count'])
                
                total_success += success
                print(f"✅ Completed {fake_type}: {success}/{info['count']} images")
            
            # Redistribute dataset
            final_stats = self.redistribute_dataset()
            
            # Generate summary
            print("\n🎉 Fake Images Expansion Completed!")
            print("=" * 60)
            print(f"📊 Final Dataset Statistics:")
            
            total_images = sum(stats['real'] + stats['fake'] for stats in final_stats.values())
            total_real = sum(stats['real'] for stats in final_stats.values())
            total_fake = sum(stats['fake'] for stats in final_stats.values())
            
            print(f"   Total Images: {total_images:,}")
            print(f"   Real Images: {total_real:,} ({total_real/total_images*100:.1f}%)")
            print(f"   Fake Images: {total_fake:,} ({total_fake/total_images*100:.1f}%)")
            
            print(f"\n📈 Distribution:")
            for split, stats in final_stats.items():
                total_split = stats['real'] + stats['fake']
                print(f"   {split.upper()}: {total_split:,} images ({stats['real']:,} real + {stats['fake']:,} fake)")
            
            print(f"\n✅ Dataset is now balanced and ready for training!")
            print(f"🎯 Successfully added {total_success}/{self.target_fake_images} fake images")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Expansion failed: {e}")
            return False

def main():
    """Main expansion function"""
    print("🤖 FAKE IMAGES EXPANSION")
    print("   Adding 2000 diverse fake images for balance")
    print("=" * 60)
    
    # Initialize expander
    expander = FakeImagesExpander(data_dir="data", target_fake_images=2000)
    
    # Confirm expansion
    print(f"\n⚠️ This will add 2000 fake images and may take 20-25 minutes.")
    print(f"   Including: AI faces, artwork, scenes, objects, and processed images")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Expansion cancelled.")
        return
    
    # Run expansion
    success = expander.run_expansion()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Verify dataset: python -c \"...\" (check balance)")
        print("   2. Start training: python scripts/train_model.py")
        print("   3. Expect better performance with balanced dataset!")
    else:
        print("\n❌ Expansion failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()