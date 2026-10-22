import numpy as np
import pytest
from musicmix.audio import AudioTrack
from musicmix.aesthetic_critic import seam_excerpt


def test_seam_is_in_the_middle_of_real_critic_context():
    rate=48000;samples=np.zeros((rate*20,2),dtype=np.float32);samples[rate*10]=1
    track=AudioTrack('fixture','fixture',rate,samples)
    mono,meta=seam_excerpt(track,rate*10)
    assert len(mono)==160000 and meta['source_start_sample']==rate*5
    assert meta['source_end_sample']==rate*15 and meta['seam_seconds_in_excerpt']==5
    assert abs(np.argmax(mono)-80000)<=1


def test_critic_never_silently_pads_a_seam_near_source_edges():
    track=AudioTrack('short','fixture',48000,np.zeros((48000*8,2),dtype=np.float32))
    with pytest.raises(ValueError,match='Insufficient real context'):
        seam_excerpt(track,48000*4)
