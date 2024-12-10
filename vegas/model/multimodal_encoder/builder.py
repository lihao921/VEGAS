import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower, LanguageBindAudioTower

# ============================================================================================================
cache_dir='/cache_dir'
def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        image_tower = '/cache_dir/LanguageBind_Image'
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        image_tower = '/cache_dir/LanguageBind_Image'
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir=cache_dir, **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        video_tower = '/cache_dir/LanguageBind_Video_merge'  

        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir=cache_dir, **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================


def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    if audio_tower.endswith('LanguageBind_Audio'):
        audio_tower = '/cache_dir/LanguageBind_Audio'
        return LanguageBindAudioTower(audio_tower, args=audio_tower_cfg, cache_dir=cache_dir, **kwargs)
    raise ValueError(f'Unknown audio tower: {audio_tower}')
# ============================================================================================================