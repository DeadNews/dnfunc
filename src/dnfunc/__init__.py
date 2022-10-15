#!/usr/bin/env python
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePath
from shutil import which
from typing import Any, NamedTuple, TypeVar, Union

import havsfunc as hav
import insane_aa as iaa
from kagefunc import adaptive_grain, kirsch, retinex_edgemask
from lvsfunc import rfs
from lvsfunc.types import Range
from vapoursynth import GRAY16, RGB, YUV420P10, VideoFrame, VideoNode, core
from vsutil import depth, get_depth, get_y, iterate, join, split
from yaml import safe_load

T = TypeVar("T")
Maps = Union[Range, list[Range]]
RangeNormalized = Union[int, tuple[int, int]]
VideoFunc1 = Callable[[VideoNode], VideoNode]
VideoFunc2 = Callable[[VideoNode, VideoNode], VideoNode]

PROC_DEPTH = 16  # processing_depth


def load_yaml(file_path: str) -> dict | None:
    """
    Load yaml settings
    """
    f1 = Path(file_path)

    return safe_load(f1.read_text()) if f1.is_file() else None


def override_dc(data_class: T, block: str, zone: str = "", **override: Any) -> T:
    """
    Override default data_class params

    Configuration preference order:
    1. func params
    2. yaml zone
    3. yaml main
    4. default
    """
    settings = load_yaml("./settings.yaml")

    if settings is not None:
        block_settings = settings.get(block)

        if block_settings is not None:
            # yaml main
            data_class = replace(data_class, **block_settings["main"])

            if zone and zone != "main":
                # yaml zone
                data_class = replace(data_class, **block_settings[zone])

    # func params
    return replace(data_class, **override)


######
# aa #
######


class NumFramesError(Exception):
    pass


class Edi3Mode(NamedTuple):
    eedi3_mode: iaa.EEDI3Mode
    device: int
    nnedi3_mode: iaa.NNEDI3Mode


def get_edi3_mode() -> Edi3Mode:
    if which("nvidia-smi") is not None:
        return Edi3Mode(
            eedi3_mode=iaa.EEDI3Mode.OPENCL,
            device=0,
            nnedi3_mode=iaa.NNEDI3Mode.NNEDI3CL,
        )
    else:
        return Edi3Mode(
            eedi3_mode=iaa.EEDI3Mode.CPU,
            device=-1,
            nnedi3_mode=iaa.NNEDI3Mode.ZNEDI3,
        )


@dataclass(frozen=True)
class AASettings:
    desc_h: int = 0
    desc_str: float = 0.32
    kernel: str = "bicubic"
    bic_b: float = 1 / 3
    bic_c: float = 1 / 3
    taps: int = 0
    #
    eedi3_only: bool = False
    #
    nrad: int = 2
    alpha: float = 0.2
    beta: float = 0.25
    gamma: int = 1000
    #
    resc_bc: bool = False
    resc_mthr: int = 40
    resc_expr: str = ""
    #
    uv_desc_h: int = 0
    uv_desc_str: float = 0.32


def insane_aa(
    clip: VideoNode,
    aaset: AASettings,
    out_mode: iaa.ClipMode,
    desc_h: int,
    desc_str: float,
    ext_mask: VideoNode | None = None,
) -> VideoNode:

    edi3set = get_edi3_mode()
    return iaa.insaneAA(
        clip=clip,
        eedi3_mode=edi3set.eedi3_mode,
        eedi3_device=edi3set.device,
        nnedi3_mode=edi3set.nnedi3_mode,
        nnedi3_device=edi3set.device,
        descale_strength=desc_str,
        descale_height=desc_h,
        kernel=aaset.kernel,
        bicubic_b=aaset.bic_b,
        bicubic_c=aaset.bic_c,
        lanczos_taps=aaset.taps,
        nns=4,
        nsize=4,
        mdis=20,
        nrad=aaset.nrad,
        alpha=aaset.alpha,
        beta=aaset.beta,
        gamma=aaset.gamma,
        external_mask=ext_mask,
        output_mode=out_mode,
    )


def aa_yuv(clip: VideoNode, aaset: AASettings) -> VideoNode:
    planes = split(clip)

    planes[0] = insane_aa(
        clip=planes[0],
        aaset=aaset,
        out_mode=iaa.ClipMode.MASKED,
        desc_str=aaset.desc_str,
        desc_h=aaset.desc_h,
    )
    planes[1] = insane_aa(
        clip=planes[1],
        aaset=aaset,
        out_mode=iaa.ClipMode.MASKED,
        desc_str=aaset.uv_desc_str,
        desc_h=aaset.uv_desc_h,
    )
    planes[2] = insane_aa(
        clip=planes[2],
        aaset=aaset,
        out_mode=iaa.ClipMode.MASKED,
        desc_str=aaset.uv_desc_str,
        desc_h=aaset.uv_desc_h,
    )

    return join(planes)


def aa(
    clip: VideoNode,
    zone: str = "",
    epname: str = "",
    ext_mask: VideoNode | None = None,
    **override: Any,
) -> VideoNode:
    """
    Anti-aliasing wrapper
    """
    if epname:
        f1 = Path(f"./temp/{epname}_aa_lossless.mp4")
        if f1.is_file():
            aa_lossless = source(f1)

            if aa_lossless.num_frames != clip.num_frames:
                raise NumFramesError(f"{aa_lossless.num_frames=}, {clip.num_frames=}")

            return aa_lossless

    aaset = AASettings()
    aaset = override_dc(aaset, block="aa", zone=zone, **override)

    if aaset.uv_desc_h:
        return aa_yuv(clip=clip, aaset=aaset)
    else:
        return insane_aa(
            clip=clip,
            aaset=aaset,
            out_mode=iaa.ClipMode.FULL,
            desc_str=aaset.desc_str,
            desc_h=clip.height if aaset.eedi3_only else aaset.desc_h,
            ext_mask=ext_mask,
        )


def diff_mask(clipa: VideoNode, clipb: VideoNode, mthr: int = 25) -> VideoNode:
    return (
        core.std.MakeDiff(clipa=get_y(clipa), clipb=get_y(clipb), planes=[0])
        .std.Prewitt()
        .std.Expr(f"x {mthr} < 0 x ?")
        .std.Convolution([1] * 9)
        .std.Convolution([1] * 9)
        .std.Expr("x 8 - 2.2 *")
    )


def save_titles(
    oped_clip: VideoNode,
    ncoped: VideoNode,
    ncoped_aa: VideoNode,
) -> VideoNode:
    """
    Save OP/ED titles with expr and diff_mask
    """
    oped_planes = split(oped_clip)

    oped_planes[0] = core.std.Expr(
        [oped_planes[0], get_y(ncoped), get_y(ncoped_aa)], ["x y - z +"]
    )

    saved_titles = join(oped_planes)
    mask = diff_mask(oped_clip, ncoped)

    return masked_merge(saved_titles, oped_clip, mask=mask, yuv=False)


def oped(
    clip: VideoNode,
    name: str,
    offset: int,
    start: int,
    end: int,
    zone: str = "",
    ext_nc: VideoNode | None = None,
    input_aa: bool = False,
    edgefix: VideoFunc1 | None = None,
    filtr: VideoFunc2 | None = None,
) -> VideoNode:
    """
    Save OP/ED titles wrapper
    """
    ncoped_end = end - 1 - start + offset

    oped_clip = clip.std.Trim(start, end - 1)
    ncoped = source(f"./in/{name}.mp4").std.Trim(offset, ncoped_end)

    ncoped_aa_src = ext_nc.std.Trim(offset, ncoped_end) if ext_nc else ncoped

    f1 = Path(f"./temp/{name}_aa_lossless.mp4")
    if f1.is_file():
        ncoped_aa = source(f1).std.Trim(offset, ncoped_end)
    elif edgefix:
        ncoped_ef = edgefix(ncoped_aa_src)
        ncoped_aa = aa(ncoped_ef, zone=zone)
    else:
        ncoped_aa = aa(ncoped_aa_src, zone=zone)

    if filtr:
        if input_aa:
            ncoped = ncoped_aa
            # ↑ if you want to keep the AA-Titles
        f2 = Path(f"./temp/{name}_filt_lossless.mp4")
        if f2.is_file():
            ncoped_aa = source(f2).std.Trim(offset, ncoped_end)
        else:
            ncoped_aa = filtr(ncoped_aa, ncoped)

    if not (oped_clip.num_frames == ncoped.num_frames == ncoped_aa.num_frames):
        raise NumFramesError(
            f"{oped_clip.num_frames=}, {ncoped.num_frames=}, {ncoped_aa.num_frames=}"
        )

    return save_titles(oped_clip=oped_clip, ncoped=ncoped, ncoped_aa=ncoped_aa)


##########
# filter #
##########


def gradfun_mask(source: VideoNode, thr_det: float = 1, mode: int = 3) -> VideoNode:
    """
    Stolen from fvsfunc
    """
    from muvsfunc import _Build_gf3_range_mask as gf3_range_mask  # noqa: N813

    src_y = get_y(source)
    src_8 = depth(src_y, 8)

    tl = max(thr_det * 0.75, 1.0) - 0.0001
    th = max(thr_det, 1.0) + 0.0001
    mexpr = f"x {tl} - {th} {tl} - / 255 *"

    if mode > 0:
        deband_mask = gf3_range_mask(src_8, radius=mode).std.Expr(mexpr).rgvs.RemoveGrain(22)
        if mode > 1:
            deband_mask = deband_mask.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])
            if mode > 2:
                deband_mask = deband_mask.std.Convolution([1] * 9)

        return depth(deband_mask, PROC_DEPTH)


def adaptive_mix(
    clip: VideoNode,
    f1: VideoNode,
    f2: VideoNode,
    scaling: float,
    yuv: bool = False,
) -> VideoNode:

    adaptive_mask = adaptive_grain(clip=clip, luma_scaling=scaling, show_mask=True).std.Invert()

    return masked_merge(f1, f2, mask=adaptive_mask, yuv=yuv)


def adaptive_debandmask(
    clip: VideoNode,
    source: VideoNode,
    db_mask: VideoNode,
    yaml: dict,
    db_expr: str = "",
) -> VideoNode:

    for zone in yaml:
        db_mode = yaml[zone]["db_mode"]
        thr_det = yaml[zone]["db_thr"]
        scaling = yaml[zone]["scaling"]

        db_n = gradfun_mask(source=source, thr_det=thr_det, mode=db_mode)
        db_mask = adaptive_mix(
            clip=clip,
            f1=db_mask,
            f2=db_n,
            scaling=scaling,
            yuv=False,
        )

    if db_expr:
        db_mask = db_mask.std.Expr(db_expr)

    return db_mask


def adaptive_smdegrain(clip: VideoNode, smdegrain: VideoNode, yaml: dict) -> VideoNode:

    for zone in yaml:
        sm_thr = yaml[zone]["sm_thr"]
        sm_pref_mode = yaml[zone]["sm_pref_mode"]
        scaling = yaml[zone]["scaling"]

        dn_n = smdegrain_(clip=clip, sm_thr=sm_thr, sm_pref_mode=sm_pref_mode)

        smdegrain = adaptive_mix(clip=clip, f1=smdegrain, f2=dn_n, scaling=scaling, yuv=True)

    return smdegrain


def save_uv_unique_lines(
    clip: VideoNode,
    source: VideoNode,
    mode: str = "retinex",
    sigma: float = 1.0,
) -> VideoNode:

    clip_planes = split(clip)
    src_planes = split(source)

    if mode == "retinex":
        mask_u = retinex_edgemask(src_planes[1], sigma=sigma)
        mask_v = retinex_edgemask(src_planes[2], sigma=sigma)
    elif mode == "kirsch":
        mask_u = edge_detect(src_planes[1], mode="kirsch")
        mask_v = edge_detect(src_planes[2], mode="kirsch")

    fix_u = masked_merge(clip_planes[1], src_planes[1], mask_u)
    fix_v = masked_merge(clip_planes[2], src_planes[2], mask_v)

    return join([clip_planes[0], fix_u, fix_v])


def save_black(
    clip: VideoNode,
    filtered: VideoNode,
    threshold: float = 0.06276,
) -> VideoNode:
    """
    stolen from https://github.com/Irrational-Encoding-Wizardry/lvsfunc/blob/master/lvsfunc.py
    return filtered when avg exceeds the threshold
    """

    def _diff(
        n: int, f: VideoFrame, clip: VideoNode, filtered: VideoNode, threshold: float  # noqa: U100
    ) -> VideoNode:
        return filtered if f.props.PlaneStatsAverage > threshold else clip

    return core.std.FrameEval(
        clip=clip,
        eval=partial(_diff, clip=clip, filtered=filtered, threshold=threshold),
        prop_src=core.std.PlaneStats(clip),
    )


def f3kdb_deband(
    clip: VideoNode,
    det_y: int,
    grainy: int,
    drange: int,
    yuv: bool = False,
) -> VideoNode:
    """
    https://f3kdb.readthedocs.io/en/latest/presets.html
    """
    return core.f3kdb.Deband(
        clip=clip,
        range=drange,
        dither_algo=3,
        y=det_y,
        cb=det_y if yuv else 0,
        cr=det_y if yuv else 0,
        blur_first=True,
        grainy=grainy,
        grainc=grainy / 2 if yuv else 0,
        dynamic_grain=False,
        keep_tv_range=True,
        output_depth=PROC_DEPTH,
    )


def contrasharp(
    clip: VideoNode,
    source: VideoNode,
    cs_mask: VideoNode,
    sm_thr: int,
    cs_val: float,
) -> tuple[VideoNode, VideoNode]:

    from CSMOD import CSMOD

    contrasharped = CSMOD(
        clip,
        source=source,
        preset="detail",
        edgemask=cs_mask,
        thSAD=sm_thr,
    )
    clip_expr = core.std.Expr([contrasharped, clip], f"x {cs_val} * y 1 {cs_val} - * +")

    return (clip_expr, contrasharped)


def _out_mask(mask: VideoNode) -> VideoNode:
    return core.resize.Point(mask, format=YUV420P10, matrix_s="709")


def smdegrain_(clip: VideoNode, sm_thr: int = 48, sm_pref_mode: int = 1) -> VideoNode:

    sm_set = {
        "prefilter": sm_pref_mode,
        "tr": 4,
        "RefineMotion": True,
        "contrasharp": False,
    }

    if isinstance(sm_thr, int):
        return hav.SMDegrain(clip, thSAD=sm_thr, plane=4, chroma=True, **sm_set)

    elif isinstance(sm_thr, list):
        while len(sm_thr) < 3:
            sm_thr.append(sm_thr[len(sm_thr) - 1])

        if clip.format.num_planes == 1:
            return hav.SMDegrain(clip, thSAD=sm_thr[0], **sm_set)

        planes = split(clip)

        planes[0] = hav.SMDegrain(planes[0], thSAD=sm_thr[0], **sm_set)
        planes[1] = hav.SMDegrain(planes[1], thSAD=sm_thr[1], **sm_set)
        planes[2] = hav.SMDegrain(planes[2], thSAD=sm_thr[2], **sm_set)

        return join(planes)


def bm3d_(
    clip: VideoNode,
    bm_sigma: float = 2,
    bm_radius: float = 1,
    sm_thr: int = 48,
    sm_pref_mode: int = 1,
) -> VideoNode:

    from mvsfunc import BM3D

    planes = split(clip)

    planes[0] = BM3D(planes[0], sigma=bm_sigma, radius1=bm_radius)
    planes[1] = smdegrain_(planes[1], sm_thr=sm_thr, sm_pref_mode=sm_pref_mode)
    planes[2] = smdegrain_(planes[2], sm_thr=sm_thr, sm_pref_mode=sm_pref_mode)

    return join(planes)


@dataclass(frozen=True)
class FilterSettings:
    rt_sigma: float = 1.0
    dn_mode: str = "smdegrain"
    dn_ttsmooth: bool = False
    bm_sigma: float = 2.0
    bm_radius: int = 1
    sm_thr: int = 40
    sm_pref_mode: int = 1
    dn_pref: bool = False
    dn_pref_scaling: float = 0.0
    dn_pref_mul: int = 0
    dn_save_uv: bool = False
    dn_adaptive: dict | None = None
    dn_expr: str = ""
    cs_mode: int = 1
    cs_val: float = 0.5
    cs_merge: int = 0
    db_thr: float = 2.1
    db_mode: int = 3
    db_gf_mode: int = 2
    db_rt_mode: int = 2
    db_pref: bool = False
    db_det: int = 64
    db_grain: int = 48
    db_range: int = 15
    db_yuv: bool = False
    db_saveblack: int = 1
    db_saveblack_tolerance: int = 2
    db_adaptive: dict | None = None
    db_expr: str = ""
    ag_str: float = 0.0
    ag_scaling: float = 24.0
    ag_saveblack: int = 1
    ag_saveblack_tolerance: int = 2


def filt(  # noqa: C901
    mrgc: VideoNode,
    zone: str = "",
    out_mode: int = 0,
    prefilt_func: VideoFunc2 | None = None,
    **override: Any,
) -> VideoNode:

    fset = FilterSettings()
    fset = override_dc(fset, block="filt", zone=zone, **override)

    clip16 = depth(mrgc, PROC_DEPTH)

    if prefilt_func:
        clip16 = prefilt_func(mrgc, clip16)

    rt_mask_clip16 = retinex_edgemask(src=clip16, sigma=fset.rt_sigma)

    if fset.dn_mode is None:
        denoised = clip16
    else:
        # dn_ttsmooth
        if fset.dn_ttsmooth:
            ttsmooth_set = {"thresh": 1, "mdiff": 0, "strength": 1}

            ttmpsm = clip16.ttmpsm.TTempSmooth(maxr=7, fp=True, **ttsmooth_set)
            src_denoise = masked_merge(ttmpsm, clip16, mask=rt_mask_clip16, yuv=True)
        else:
            src_denoise = clip16

        # denoised
        if fset.dn_mode == "smdegrain":
            full_denoise = smdegrain_(
                clip=src_denoise,
                sm_thr=fset.sm_thr,
                sm_pref_mode=fset.sm_pref_mode,
            )
            if fset.dn_adaptive is not None:
                full_denoise = adaptive_smdegrain(
                    clip=clip16,
                    smdegrain=full_denoise,
                    yaml=fset.dn_adaptive,
                )

        elif fset.dn_mode == "bm3d":
            full_denoise = bm3d_(
                src_denoise,
                bm_sigma=fset.bm_sigma,
                bm_radius=fset.bm_radius,
                sm_thr=fset.sm_thr,
                sm_pref_mode=fset.sm_pref_mode,
            )

        denoised = masked_merge(full_denoise, clip16, mask=rt_mask_clip16, yuv=True)

        # dn_pref
        if fset.dn_pref or fset.cs_mode:
            rt_mask_denoised = retinex_edgemask(denoised, sigma=fset.rt_sigma)

            if fset.dn_expr:
                rt_mask_denoised_def = rt_mask_denoised
                rt_mask_denoised = rt_mask_denoised.std.Expr(fset.dn_expr)

            rt_mask_mix = (
                adaptive_mix(
                    clip=clip16,
                    f1=rt_mask_clip16,
                    f2=rt_mask_denoised,
                    scaling=fset.dn_pref_scaling,
                    yuv=False,
                )
                if fset.dn_pref_scaling
                else rt_mask_denoised
            )

            if fset.dn_pref_mul:
                rt_mask_denoised_x2 = smdegrain_(
                    clip=rt_mask_denoised,
                    sm_thr=fset.sm_thr * fset.dn_pref_mul,
                    sm_pref_mode=fset.sm_pref_mode,
                )
                if fset.db_rt_mode == 4 and fset.db_adaptive is not None:
                    rt_mask_denoised_mix = adaptive_mix(
                        clip=clip16,
                        f1=rt_mask_denoised,
                        f2=rt_mask_denoised_x2,
                        scaling=fset.db_adaptive["z2"]["scaling"],
                        yuv=False,
                    )

            if fset.dn_pref or fset.db_gf_mode == 3:
                denoised_pref = masked_merge(full_denoise, clip16, mask=rt_mask_mix, yuv=True)

        if fset.dn_pref:
            denoised = denoised_pref

        # dn_save_uv
        if fset.dn_save_uv:
            denoised = save_uv_unique_lines(clip=denoised, source=clip16, sigma=fset.rt_sigma)

        # contrasharp
        if fset.cs_mode:
            if fset.cs_mode == 1:  # default
                cs_mask = rt_mask_denoised
            elif fset.cs_mode == 2:
                cs_mask = rt_mask_denoised_x2
            elif fset.cs_mode == 3:
                cs_mask = rt_mask_mix
            elif fset.cs_mode == 4:
                cs_mask = rt_mask_denoised_def

            denoised_expr, denoised_fullcs = contrasharp(
                clip=denoised,
                source=clip16,
                cs_mask=cs_mask,
                sm_thr=fset.sm_thr,
                cs_val=fset.cs_val,
            )
            if not fset.cs_merge:
                denoised = denoised_expr
            else:
                if fset.cs_merge == 1:
                    cs_mask_merge = cs_mask
                elif fset.cs_merge == 2:
                    cs_mask_merge = cs_mask.std.Inflate()
                denoised = masked_merge(denoised, denoised_expr, mask=cs_mask_merge, yuv=True)

    if fset.db_thr == 0:
        debanded = denoised
    else:
        debanded = f3kdb_deband(
            clip=denoised,
            det_y=fset.db_det,
            grainy=fset.db_grain,
            drange=fset.db_range,
            yuv=fset.db_yuv,
        )
        if fset.db_gf_mode == 1:
            gradfun_src = denoised_fullcs
        elif fset.db_gf_mode == 2:  # default
            gradfun_src = denoised
        elif fset.db_gf_mode == 3:
            gradfun_src = denoised_pref
        elif fset.db_gf_mode == 4:
            gradfun_src = full_denoise

        db_mask = gradfun_mask(source=gradfun_src, thr_det=fset.db_thr, mode=fset.db_mode)

        if out_mode == 2:
            db_mask_gradfun = db_mask

        if fset.db_adaptive:
            db_mask = adaptive_debandmask(
                clip=clip16,
                source=gradfun_src,
                db_mask=db_mask,
                yaml=fset.db_adaptive,
                db_expr=fset.db_expr,
            )

        if fset.db_saveblack == 1:
            debanded = save_black(clip=denoised, filtered=debanded, threshold=0.06276)
        elif fset.db_saveblack == 2:
            black_mask = rfs_color(
                mask_src=mrgc,
                out_mask=True,
                tolerance=fset.db_saveblack_tolerance,
            )
            db_mask = masked_merge(db_mask, black_mask, mask=black_mask, yuv=False)

        debanded = masked_merge(debanded, denoised, mask=db_mask, yuv=fset.db_yuv)

    # final masked_merge
    if fset.dn_mode is None:
        filtered = debanded
    else:
        if fset.db_rt_mode == 1:
            rt_mask_afterdb = rt_mask_clip16
        elif fset.db_rt_mode == 2:  # default
            rt_mask_afterdb = rt_mask_mix
        elif fset.db_rt_mode == 3:
            rt_mask_afterdb = rt_mask_denoised
        elif fset.db_rt_mode == 4:
            rt_mask_afterdb = rt_mask_denoised_mix
        elif fset.db_rt_mode == 5:
            rt_mask_afterdb = rt_mask_denoised_x2

        if fset.db_pref:
            filtered = masked_merge(debanded, denoised, mask=rt_mask_afterdb, yuv=fset.db_yuv)
        else:
            filtered = masked_merge(debanded, clip16, mask=rt_mask_afterdb, yuv=fset.db_yuv)

    # adaptive_grain
    if out_mode == 1:
        ag_mask = adaptive_grain(filtered, luma_scaling=fset.ag_scaling, show_mask=True)
    if fset.ag_str != 0:
        grained = adaptive_grain(filtered, luma_scaling=fset.ag_scaling, strength=fset.ag_str)

        if fset.ag_saveblack == 1:
            filtered = save_black(filtered, grained, threshold=0.06276)
        elif fset.ag_saveblack == 2:
            filtered = rfs_color(
                f1=grained,
                f2=filtered,
                mask_src=mrgc,
                tolerance=fset.ag_saveblack_tolerance,
            )
        else:
            filtered = grained

    if out_mode == 0:
        return filtered
    elif out_mode == 1:
        return _out_mask(ag_mask)
    elif out_mode == 2:
        return _out_mask(db_mask_gradfun)
    elif out_mode == 3:
        return _out_mask(db_mask)
    elif out_mode == 4:
        return _out_mask(rt_mask_clip16)
    elif out_mode == 5:
        return _out_mask(rt_mask_mix)
    elif out_mode == 6:
        return _out_mask(rt_mask_denoised)
    elif out_mode == 7:
        return _out_mask(rt_mask_denoised_x2)


#########
# other #
#########


def chapt(epname: str, chaptname: str, fallback: str = "") -> int | None:

    chapters = load_yaml("./chapters.yaml")

    if chapters is None:
        return None

    epchaps = chapters[epname]

    return epchaps.get(chaptname, epchaps.get(fallback))


def load_map(epname: str, mapname: str) -> Any:

    maps = load_yaml("./maps.yaml")

    return None if maps is None else maps[mapname].get(epname)


def fname(file: str, aa_mode: bool = False) -> str:
    f1 = PurePath(file)

    return f1.stem[:-3] if aa_mode else f1.stem


def source(
    file: str | Path | PurePath,
    bits: int = 0,
    fpsnum: int = 0,
    fpsden: int = 0,
) -> VideoNode:

    f1 = PurePath(file)

    if f1.suffix == ".mp4":
        src = core.lsmas.LibavSMASHSource(source=f1)
    else:
        src = core.lsmas.LWLibavSource(source=f1)

    if fpsnum and fpsden:
        src = src.std.AssumeFPS(fpsnum=fpsnum, fpsden=fpsden)

    if not bits:
        bits = PROC_DEPTH

    return depth(src, bits)


def average(clips: list[VideoNode]) -> VideoNode:
    min_num_frames = min(clip.num_frames for clip in clips)

    return core.average.Mean([clip.std.Trim(0, min_num_frames - 1) for clip in clips])


def out(
    clip: VideoNode,
    epname: str,
    bits: int = 10,
    filtred: VideoFunc2 | None = None,
) -> VideoNode:

    if get_depth(clip) != bits:
        clip = depth(clip, bits)

    f1 = Path(f"./temp/{epname}_lossless.mp4")

    if f1.is_file():
        lossless = source(f1, bits=bits)
        if filtred:
            lossless = filtred(clip, lossless)

        return lossless
    else:
        return clip


def pw(
    mrgc: VideoNode,
    masks: Iterable[int] = (3, 4),
    mask_zone: str = "",
    epis: VideoNode = None,
    clip: VideoNode = None,
    clip2_zone: str = "",
    ext_rip: VideoNode | None = None,
) -> VideoNode:

    pw_list = []
    if masks:
        for out_mode in masks:
            mask = filt(mrgc, zone=mask_zone, out_mode=out_mode)
            pw_list.append(mask)

    if epis:
        pw_list.append(depth(epis, 10))
    if clip:
        pw_list.append(clip)
    if clip2_zone:
        pw_list.append(depth(filt(mrgc, zone=clip2_zone), 10))
    if ext_rip:
        pw_list.append(ext_rip)

    return core.std.Interleave(pw_list)


def aa_pw(epis: VideoNode, zones: Iterable[str] = ("main", "test")) -> VideoNode:

    pw_list = []
    if epis:
        pw_list.append(epis)

    for zone in zones:
        aaep = aa(epis, zone=zone)
        pw_list.append(aaep)

    return core.std.Interleave(pw_list)


def masked_merge(
    f1: VideoNode,
    f2: VideoNode,
    mask: VideoNode,
    yuv: bool = False,
) -> VideoNode:
    return core.std.MaskedMerge(
        clipa=f1,
        clipb=f2,
        mask=mask,
        planes=[0, 1, 2] if yuv else [0],
    )


def check_num_frames(epis: VideoNode, clip: VideoNode) -> None:
    if epis.num_frames != clip.num_frames:
        raise NumFramesError(f"{epis.num_frames=}, {clip.num_frames=}")


def _mask_resize(mask: VideoNode, format_src: VideoNode | None = None) -> VideoNode:  # noqa: U100

    # if format_src:
    #     mask_format = format_src.format.replace(color_family=GRAY, subsampling_w=0, subsampling_h=0)
    # else:
    #     mask_format = GRAY16

    mask_format = GRAY16

    return (
        mask.resize.Point(matrix_s="709", format=mask_format)
        if mask.format.color_family == RGB
        else mask.resize.Point(format=mask_format)
    )


def color_mask(
    mask_src: VideoNode,
    format_src: VideoNode,
    color: str = "$000000",
    tolerance: int = 2,
) -> VideoNode:

    if get_depth(mask_src) != 8:
        mask_src = depth(mask_src, 8)

    mask = mask_src.tcm.TColorMask(colors=color, tolerance=tolerance)

    return _mask_resize(mask, format_src)


def rfs_color(
    mask_src: VideoNode,
    f1: VideoNode | None = None,
    f2: VideoNode | None = None,
    color: str = "$000000",
    tolerance: int = 2,
    out_mask: bool = False,
    maps: Maps | None = None,
) -> VideoNode:

    mask = color_mask(mask_src=mask_src, format_src=f1, color=color, tolerance=tolerance)
    if out_mask:
        return mask

    replaced = masked_merge(f1, f2, mask=mask, yuv=False)

    return rfs(f1, replaced, maps) if maps else replaced


def image_mask(maskname: str, format_src: VideoNode) -> VideoNode:
    mask = core.imwri.Read(f"./mask/{maskname}.png")
    return _mask_resize(mask, format_src)


def rfs_image(
    mrgc: VideoNode,
    epis: VideoNode,
    maskname: str = "",
    yuv: bool = False,
    maps: Maps | None = None,
) -> VideoNode:

    mask = image_mask(maskname, format_src=mrgc)
    replaced = masked_merge(mrgc, epis, mask=mask, yuv=yuv)

    return rfs(mrgc, replaced, maps) if maps else replaced


def rfs_diff(
    mrgc: VideoNode,
    epis: VideoNode,
    maps: Maps | None = None,
    mthr: int = 25,
    yuv: bool = True,
    out_mask: bool = False,
) -> VideoNode:

    mask = diff_mask(mrgc, epis, mthr=mthr)
    if out_mask:
        return mask

    replaced = masked_merge(mrgc, epis, mask=mask, yuv=yuv)

    return rfs(mrgc, replaced, maps) if maps else replaced


def diff_rescale_mask(source: VideoNode, dset: AASettings) -> VideoNode:
    """
    Builds mask from difference of original and re-upscales clips;
    based on atomchtools
    """
    from descale import Descale
    from fvsfunc import Resize

    clip = get_y(source) if source.format.num_planes != 1 else source

    desc = Descale(
        src=clip,
        width=hav.m4((source.width * dset.desc_h) / source.height),
        height=dset.desc_h,
        kernel=dset.kernel,
        b=dset.bic_b,
        c=dset.bic_c,
        taps=dset.taps,
    )

    upsc_set = (
        {"kernel": dset.kernel, "a1": dset.bic_b, "a2": dset.bic_c, "taps": dset.taps}
        if dset.resc_bc
        else {"kernel": dset.kernel, "a1": 1 / 3, "a2": 1 / 3}
    )
    upsc = Resize(src=desc, w=source.width, h=source.height, **upsc_set)

    if get_depth(source) != 8:
        clip = depth(clip, 8)
        upsc = depth(upsc, 8)

    mask = (
        core.std.MakeDiff(clip, upsc)
        .rgvs.RemoveGrain(2)
        .rgvs.RemoveGrain(2)
        .hist.Luma()
        .std.Expr(f"x {dset.resc_mthr} < 0 x ?")
        .std.Prewitt()
        .std.Maximum()
        .std.Maximum()
        .std.Deflate()
    )

    if dset.resc_expr:
        mask = mask.std.Expr(dset.resc_expr)

    if get_depth(mask) != (source_depth := get_depth(source)):
        mask = depth(mask, source_depth)

    return mask


def rfs_resc(
    mrgc: VideoNode = None,
    epis: VideoNode = None,
    zone: str = "",
    maps: Maps | None = None,
    filt: VideoFunc1 = None,
    out_mask: bool = False,
    **override: Any,
) -> VideoNode:

    dset = AASettings()
    dset = override_dc(dset, block="aa", zone=zone, **override)

    mask = diff_rescale_mask(epis, dset=dset)
    if filt:
        mask = filt(mask)
    if out_mask:
        return mask

    replaced = masked_merge(mrgc, epis, mask=mask)

    return rfs(mrgc, replaced, maps) if maps else replaced


def _hard(clip: VideoNode, mthr: int, yuv: bool = False) -> VideoNode:
    from HardAA import HardAA

    return HardAA(
        clip=clip,
        mask="simple",
        mthr=mthr,
        LumaOnly=not yuv,
        useCL=get_edi3_mode().device != -1,
    )


def hard(clip: VideoNode, mthr: int, zone: str = "", **override: Any) -> VideoNode:

    yuv = yuv if (yuv := override.get("yuv")) is not None else False

    if zone is None and override.get("desc_h") is None:
        return _hard(clip, mthr=mthr, yuv=yuv)

    revset = AASettings()
    revset = override_dc(revset, block="aa", zone=zone, **override)

    revert = iaa.revert_upscale(
        clip=clip,
        descale_strength=revset.desc_str,
        descale_height=revset.desc_h,
        kernel=revset.kernel,
        bic_b=revset.bic_b,
        bic_c=revset.bic_c,
    )

    hard_aa = _hard(revert, mthr=mthr, yuv=yuv)
    return rescale_(hard_aa, mode="insane_aa", ref_clip=clip)


def rfs_hard(
    mrgc: VideoNode,
    src: VideoNode,
    mthr: int,
    maps: Maps,
    zone: str = "",
    **override: Any,
) -> VideoNode:

    hard_ = hard(src, mthr=mthr, zone=zone, **override)

    return rfs(mrgc, hard_, maps)


@dataclass(frozen=True)
class QTGMCSettings:
    k: float = 1
    thsad1: float = 640
    thsad2: float = 256
    thscd1: float = 180
    thscd2: float = 98
    #
    input_type: int = 1
    preset: str = "placebo"
    match_enhance: float = 0.95
    sharp: float = 0.2


def qtgmc(clip: VideoNode, zone: str = "", **override: Any) -> VideoNode:
    """
    InputType — 0 for interlaced input. Mode 1 is for general progressive material.
    Modes 2 & 3 are designed for badly deinterlaced material.
    Sharpness — The default 1.0 is fairly sharp. If using source-match the default is 0.2
    """
    qset = QTGMCSettings()
    qset = override_dc(qset, block="qtgmc", zone=zone, **override)

    device = get_edi3_mode().device

    return hav.QTGMC(
        Input=clip,
        Preset=qset.preset,
        InputType=qset.input_type,
        FPSDivisor=2 if qset.input_type == 0 else 1,
        ShutterBlur=3,
        ShowSettings=False,
        TFF=True,
        SourceMatch=3,
        MatchEnhance=qset.match_enhance,
        MatchPreset=qset.preset,
        MatchPreset2=qset.preset,
        opencl=device != -1,
        device=device,
        Sharpness=qset.sharp,
        NoiseProcess=0,
        ThSAD1=qset.thsad1 * qset.k,
        ThSAD2=qset.thsad2 * qset.k,
        ThSCD1=qset.thscd1 * qset.k,
        ThSCD2=qset.thscd2 * qset.k,
    )


def rfs_qtgmc(
    mrgc: VideoNode,
    src: VideoNode,
    maps: Maps,
    zone: str = "",
    **override: Any,
) -> VideoNode:

    stabilize = qtgmc(src, zone=zone, **override)

    return rfs(mrgc, stabilize, maps)


def get_kirsch2_mask(clip_y: VideoNode) -> VideoNode:
    n = core.std.Convolution(clip_y, [5, 5, 5, -3, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    nw = core.std.Convolution(clip_y, [5, 5, -3, 5, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    w = core.std.Convolution(clip_y, [5, -3, -3, 5, 0, -3, 5, -3, -3], divisor=3, saturate=False)
    sw = core.std.Convolution(clip_y, [-3, -3, -3, 5, 0, -3, 5, 5, -3], divisor=3, saturate=False)
    s = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, -3, 5, 5, 5], divisor=3, saturate=False)
    se = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, 5, -3, 5, 5], divisor=3, saturate=False)
    e = core.std.Convolution(clip_y, [-3, -3, 5, -3, 0, 5, -3, -3, 5], divisor=3, saturate=False)
    ne = core.std.Convolution(clip_y, [-3, 5, 5, -3, 0, 5, -3, -3, -3], divisor=3, saturate=False)
    return core.std.Expr(
        [n, nw, w, sw, s, se, e, ne],
        ["x y max z max a max b max c max d max e max"],
    )


def edge_detect(clip: VideoNode, mode: str, expr: str = "") -> VideoNode:
    clip_y = get_y(clip)

    if mode == "fine2":
        mask = hav.FineDehalo2(clip_y, showmask=1)
    elif mode == "kirsch":
        mask = kirsch(clip_y)
    elif mode == "kirsch2":
        mask = get_kirsch2_mask(clip_y)
    elif mode == "sobel":
        mask = core.std.Sobel(clip_y)

    return mask.std.Expr(expr) if expr else mask


def outerline_mask(
    clip: VideoNode,
    mode: str = "kirsch",
    max_c: int = 3,
    min_c: int = 1,
    ext_mask: VideoNode | None = None,
) -> VideoNode:

    mask = ext_mask or edge_detect(clip, mode=mode)

    mask_outer = iterate(mask, function=core.std.Maximum, count=max_c)
    mask_inner = mask.std.Inflate()
    mask_inner = iterate(mask_inner, function=core.std.Minimum, count=min_c)

    return core.std.Expr([mask_outer, mask_inner], "x y -")


@dataclass(frozen=True)
class DehaloSettings:
    rx: float = 2.0
    darkstr: float = 0
    brightstr: float = 0.5
    mode: str = "kirsch"
    max_c: int = 3
    min_c: int = 1
    mask_from_filtred: bool = True


def rfs_dehalo(
    clip: VideoNode,
    ext_dehalo: VideoNode | None = None,
    zone: str = "",
    ext_mask: VideoNode | None = None,
    maps: Maps | None = None,
    out_mask: bool = False,
    **override: Any,
) -> VideoNode:

    dehset = DehaloSettings()
    dehset = override_dc(dehset, block="dehalo", zone=zone, **override)

    if ext_dehalo:
        dehalo = ext_dehalo
        mask_src = dehalo
    else:
        dehalo = hav.DeHalo_alpha(
            clip=clip,
            rx=dehset.rx,
            ry=dehset.rx,
            darkstr=dehset.darkstr,
            brightstr=dehset.brightstr,
            lowsens=50,
            highsens=50,
            ss=1.5,
        )
        mask_src = dehalo if dehset.mask_from_filtred else clip

    mask = outerline_mask(
        mask_src,
        mode=dehset.mode,
        max_c=dehset.max_c,
        min_c=dehset.min_c,
        ext_mask=ext_mask,
    )
    if out_mask:
        return mask

    replaced = masked_merge(clip, dehalo, mask=mask, yuv=False)

    return rfs(clip, replaced, maps) if maps else replaced


def dehalo_chroma(clip: VideoNode, zone: str = "") -> VideoNode:

    planes = split(clip)
    planes[1] = rfs_dehalo(planes[1], zone=zone)
    planes[2] = rfs_dehalo(planes[2], zone=zone)

    return join(planes)


@dataclass(frozen=True)
class RepairSettings:
    edgclr_args: dict[str, int] = field(
        default_factory=lambda: ({"strength": 10, "rmode": 13, "smode": 1})
    )
    dering_args: dict[str, int] = field(
        default_factory=lambda: ({"mrad": 2, "mthr": 70, "thr": 12, "darkthr": 3})
    )
    mode: str = "kirsch"
    max_c: int = 3
    min_c: int = 1
    mask_from_filtred: bool = False


def rfs_repair(
    clip: VideoNode,
    zone: str = "",
    out_mask: bool = False,
    maps: Maps | None = None,
    **override: Any,
) -> VideoNode:
    """ """

    repset = RepairSettings()
    repset = override_dc(repset, block="repair", zone=zone, **override)

    if repset.edgclr_args:
        repair = hav.EdgeCleaner(clip, **repset.edgclr_args)
    if repset.dering_args:
        repair = hav.HQDeringmod(clip, **repset.dering_args)

    mask_src = repair if repset.mask_from_filtred else clip
    mask = outerline_mask(clip=mask_src, mode=repset.mode, max_c=repset.max_c, min_c=repset.min_c)
    if out_mask:
        return mask

    replaced = masked_merge(clip, repair, mask=mask, yuv=False)

    return rfs(clip, replaced, maps) if maps else replaced


@dataclass(frozen=True)
class LineDarkSettings:
    linedark_args: dict[str, int] | None = None


def rfs_linedark(
    clip: VideoNode,
    zone: str = "",
    maps: Maps | None = None,
    **override: Any,
) -> VideoNode:

    ldset = LineDarkSettings()
    ldset = override_dc(ldset, block="linedark", zone=zone, **override)

    replaced = hav.FastLineDarkenMOD(clip, **ldset.linedark_args)

    return rfs(clip, replaced, maps) if maps else replaced


@dataclass(frozen=True)
class SsharpSettings:
    mode: str = "cas"
    sharp: float = 0.1
    mask_mode: str = "sobel"
    mask_expr: str = ""
    yuv: bool = False


def rfs_sharp(
    clip: VideoNode,
    zone: str = "",
    maps: Maps = None,
    out_mask: bool = False,
    **override: Any,
) -> VideoNode:

    shset = SsharpSettings()
    shset = override_dc(shset, block="sharp", zone=zone, **override)

    mask = edge_detect(clip, mode=shset.mask_mode, expr=shset.mask_expr)
    if out_mask:
        return mask

    if shset.mode == "cas":
        sharp = core.cas.CAS(clip, sharpness=shset.sharp)
    elif shset.mode == "finesharp":
        from finesharp import sharpen

        sharp = sharpen(clip, sstr=sharp)

    replaced = masked_merge(clip, sharp, mask=mask, yuv=shset.yuv)

    return rfs(clip, replaced, maps) if maps else replaced


@dataclass
class EdgeFixSettings:
    crop_args: dict[str, int] | None = None
    top: int | list[int] = 0
    bottom: int | list[int] = 0
    left: int | list[int] = 0
    right: int | list[int] = 0
    radius: int | list[int] = 0
    yuv: bool = False

    @staticmethod
    def to_list(val: int | list[int]) -> list[int]:
        return val if isinstance(val, list) else [val, 0, 0]

    def check_yuv(self) -> None:
        self.yuv = any(
            isinstance(val, list)
            for val in (self.top, self.bottom, self.left, self.right, self.radius)
        )


def edgefix(
    epis: VideoNode,
    zone: str = "",
    **override: Any,
) -> tuple[VideoNode, VideoNode, VideoFunc1]:
    """ """

    def _edgefixer(epis: VideoNode) -> VideoNode:
        if not edset.yuv or not edset.crop_args:
            # luma
            return core.edgefixer.Continuity(
                epis,
                top=edset.top,
                bottom=edset.bottom,
                left=edset.left,
                right=edset.right,
                radius=edset.radius,
            )

        # yuv
        top = edset.to_list(edset.top)
        bottom = edset.to_list(edset.bottom)
        left = edset.to_list(edset.left)
        right = edset.to_list(edset.right)
        radius = edset.to_list(edset.radius)

        planes = split(epis)

        y = (
            planes[0]
            .std.Crop(**edset.crop_args)
            .edgefixer.Continuity(
                top=top[0],
                bottom=bottom[0],
                left=left[0],
                right=right[0],
                radius=radius[0],
            )
            .std.AddBorders(**edset.crop_args)
            if edset.crop_args
            else planes[0].edgefixer.Continuity(
                top=top[0],
                bottom=bottom[0],
                left=left[0],
                right=right[0],
                radius=radius[0],
            )
        )

        u = planes[1].edgefixer.Continuity(
            top=top[1],
            bottom=bottom[1],
            left=left[1],
            right=right[1],
            radius=radius[1],
        )
        v = planes[2].edgefixer.Continuity(
            top=top[2],
            bottom=bottom[2],
            left=left[2],
            right=right[2],
            radius=radius[2],
        )

        return join([y, u, v])

    edset = EdgeFixSettings()
    edset = override_dc(edset, block="edgefix", zone=zone, **override)
    edset.check_yuv()

    epis_back = epis
    epis = _edgefixer(epis)

    return (epis, epis_back, _edgefixer)


def wipe_luma_row(clip: VideoNode, **crop: Any) -> VideoNode:
    planes = split(clip)

    planes[0] = planes[0].std.Crop(**crop).std.AddBorders(**crop)

    return join(planes)


@dataclass(frozen=True)
class CropSettings:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


def crop(
    epis: VideoNode,
    zone: str = "",
    **override: Any,
) -> tuple[VideoNode, VideoFunc1]:
    """
    Crop wrapper
    """

    def _crop(epis: VideoNode) -> VideoNode:
        return core.std.CropRel(
            epis,
            top=crset.top,
            bottom=crset.bottom,
            left=crset.left,
            right=crset.right,
        )

    crset = CropSettings()
    crset = override_dc(crset, block="crop", zone=zone, **override)

    epis = _crop(epis)

    return (epis, _crop)


def to60fps_mv(clip: VideoNode) -> VideoNode:
    """
    http://avisynth.org.ru/mvtools/mvtools2.html
    """
    sup = core.mv.Super(clip, pel=2, sharp=2, rfilter=4)
    bvec = core.mv.Analyse(
        sup,
        blksize=8,
        isb=True,
        chroma=True,
        search=3,
        searchparam=1,
        truemotion=True,
        dct=1,
        overlap=4,
    )
    fvec = core.mv.Analyse(
        sup,
        blksize=8,
        isb=False,
        chroma=True,
        search=3,
        searchparam=1,
        truemotion=True,
        dct=1,
        overlap=4,
    )
    return core.mv.FlowFPS(clip, sup, bvec, fvec, num=60, den=1)


def to60fps_svp(clip: VideoNode) -> VideoNode:
    clip_p8 = depth(clip, 8) if get_depth(clip) != 8 else clip

    super_params = "{gpu: 1, pel: 2}"
    analyse_params = "{gpu: 1, block: {w:8, overlap:3}, refine: [{thsad:1000, search:{type:3}}]}"
    smoothfps_params = "{rate: {num: 5, den: 2}, algo: 2, gpuid: 0}"  # (24000/1001)*(1001/400)=60

    sup = core.svp1.Super(clip_p8, super_params)
    vectors = core.svp1.Analyse(sup["clip"], sup["data"], clip_p8, analyse_params)

    return core.svp2.SmoothFps(
        clip,
        sup["clip"],
        sup["data"],
        vectors["clip"],
        vectors["data"],
        smoothfps_params,
    )


def to60fps(clip: VideoNode, mode: str = "svp") -> VideoNode:
    if mode == "mv":
        return to60fps_mv(clip)
    elif mode == "svp":
        return to60fps_svp(clip)


def chromashift(clip: VideoNode, cx: int = 0, cy: int = 0) -> VideoNode:
    """
    cx — Horizontal chroma shift. Positive value shifts chroma to left, negative value shifts chroma to right.
    cy — Vertical chroma shift. Positive value shifts chroma upwards, negative value shifts chroma downwards.
    """
    planes = split(clip)

    planes[1] = core.resize.Spline36(planes[1], src_left=cx, src_top=cy)
    planes[2] = core.resize.Spline36(planes[2], src_left=cx, src_top=cy)

    return join(planes)


def adaptive_chromashift(
    clip: VideoNode,
    fix: VideoNode,
    pw_mode: int = 0,
) -> VideoNode:
    """
    Chromashift with comparisons for floating chromashift
    """

    def make_diff(clip: VideoNode) -> VideoNode:
        from fvsfunc import Downscale444

        desc_h = 720
        desc_w = hav.m4((clip.width * desc_h) / clip.height)
        descale = Downscale444(clip, w=desc_w, h=desc_h)

        planes_desc = split(descale)
        y = core.std.Sobel(planes_desc[0])
        u = core.std.Sobel(planes_desc[1])
        v = core.std.Sobel(planes_desc[2])

        uv = core.std.Expr([u, v], ["x y max"])

        return core.std.MakeDiff(y, uv)

    def frame_diff_eval(
        n: int,
        f: VideoFrame,
        f1: VideoNode,
        f2: VideoNode,
    ) -> VideoNode:

        c0 = f[1].props.PlaneStatsAverage > f[0].props.PlaneStatsAverage
        p1 = f[3].props.PlaneStatsAverage > f[2].props.PlaneStatsAverage
        n1 = f[5].props.PlaneStatsAverage > f[4].props.PlaneStatsAverage
        p2 = f[7].props.PlaneStatsAverage > f[6].props.PlaneStatsAverage
        n2 = f[9].props.PlaneStatsAverage > f[8].props.PlaneStatsAverage
        p3 = f[11].props.PlaneStatsAverage > f[10].props.PlaneStatsAverage
        n3 = f[13].props.PlaneStatsAverage > f[12].props.PlaneStatsAverage

        if p1 == n1:
            condition = p1 and n1
        elif p2 == n2:
            condition = p2 and n2
        elif p3 == n3:
            condition = p3 and n3
        else:
            condition = c0

        out = f2 if condition else f1

        return add_caption(n=n, f=f, out=out, condition=condition) if pw_mode else out

    def add_caption(
        n: int,
        f: VideoFrame,
        out: VideoNode,
        condition: bool,
    ) -> VideoNode:

        lines: list[str] = []
        if pw_mode == 2:
            lines.extend(
                (
                    f"Frame avg: {f[0].props.PlaneStatsAverage}",
                    f"Frame avg: {f[1].props.PlaneStatsAverage}",
                )
            )
        lines.append("fix" if condition else "clip")

        style = "Fira Code,20,&H0000FFFF,&H00000000,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1"
        return out.sub.Subtitle("\n".join(lines), start=n, end=n + 1, style=style)

    def _adaptive_chromashift(clip: VideoNode, fix: VideoNode) -> VideoNode:
        diff_def = make_diff(clip)
        diff_fix = make_diff(fix)

        s0 = core.std.PlaneStats(diff_def)  # curr0
        s1 = core.std.PlaneStats(diff_fix)  # curr0
        s2 = core.std.PlaneStats(diff_def).std.DuplicateFrames(0)  # prev1
        s3 = core.std.PlaneStats(diff_fix).std.DuplicateFrames(0)  # prev1
        s4 = core.std.PlaneStats(diff_def).std.Trim(1)  # next1
        s5 = core.std.PlaneStats(diff_fix).std.Trim(1)  # next1
        s6 = core.std.PlaneStats(diff_def).std.DuplicateFrames([0, 1])  # prev2
        s7 = core.std.PlaneStats(diff_fix).std.DuplicateFrames([0, 1])  # prev2
        s8 = core.std.PlaneStats(diff_def).std.Trim(2)  # next2
        s9 = core.std.PlaneStats(diff_fix).std.Trim(2)  # next2
        s10 = core.std.PlaneStats(diff_def).std.DuplicateFrames([0, 1, 2])  # prev3
        s11 = core.std.PlaneStats(diff_fix).std.DuplicateFrames([0, 1, 2])  # prev3
        s12 = core.std.PlaneStats(diff_def).std.Trim(3)  # next3
        s13 = core.std.PlaneStats(diff_fix).std.Trim(3)  # next3

        return core.std.FrameEval(
            clip=clip,
            eval=partial(frame_diff_eval, f1=clip, f2=fix),
            prop_src=[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13],
        )

    return _adaptive_chromashift(clip, fix)


def rescale_(
    clip: VideoNode,
    mode: str = "insane_aa",
    ref_clip: VideoNode | None = None,
    width: int = 0,
    height: int = 0,
    zone: str = "",
    **override: Any,
) -> VideoNode:

    if ref_clip is not None:
        dx = ref_clip.width
        dy = ref_clip.height
    elif width and height:
        dx = width
        dy = height
    elif height:
        dx = hav.m4((clip.width * height) / clip.height)
        dy = height
    elif width:
        dx = width
        dy = hav.m4((clip.height * width) / clip.width)

    if mode == "insane_aa":
        aaset = AASettings()
        aaset = override_dc(aaset, block="aa", zone=zone, **override)

        edi3set = get_edi3_mode()
        return iaa.rescale(
            clip,
            eedi3_mode=edi3set.eedi3_mode,
            eedi3_device=edi3set.device,
            nnedi3_mode=edi3set.nnedi3_mode,
            nnedi3_device=edi3set.device,
            dx=dx,
            dy=dy,
            nns=4,
            nsize=4,
            mdis=20,
            nrad=aaset.nrad,
            alpha=aaset.alpha,
            beta=aaset.beta,
            gamma=aaset.gamma,
        )
    elif mode == "jinc":
        return clip.jinc.JincResize(dx, dy)
    elif mode == "lanczos":
        return clip.resize.Lanczos(dx, dy, filter_param_a=2)


def downscale(clip: VideoNode, desc_h: int = 720, to420: bool = False) -> VideoNode:

    if clip.height == desc_h:
        return clip

    desc_w = hav.m4((clip.width * desc_h) / clip.height)

    if not to420:
        return clip.resize.Spline36(desc_w, desc_h)

    planes = split(clip)

    planes[0] = planes[0].resize.Spline36(desc_w, desc_h)
    planes[1] = planes[1].resize.Spline36(desc_w / 2, desc_h / 2)
    planes[2] = planes[2].resize.Spline36(desc_w / 2, desc_h / 2)

    return join(planes)


def rfs_black_crop(
    clip: VideoNode,
    maps: Maps,
    top: int = 0,
    bot: int = 0,
) -> VideoNode:

    fixed_black = clip.std.CropRel(top=top, bottom=bot).std.AddBorders(top=top, bottom=bot)

    return rfs(clip, fixed_black, maps)


def get_list(ranges: list[RangeNormalized]) -> list[int]:
    frames = []
    for x in ranges:
        if isinstance(x, tuple):
            start, end = x
            frames.extend(list(range(start, end + 1)))
        elif isinstance(x, int):
            frames.append(x)

    return frames


def pv_diff(
    tv: VideoNode,
    bd: VideoNode,
    thr: float = 72,
    name: str = "",
    exclude_ranges: list[RangeNormalized] | None = None,
) -> VideoNode:
    from lvsfunc import diff

    clips = [tv, bd]
    num_frames = [clip.num_frames for clip in clips]
    clips = [clip.std.Trim(0, min(num_frames) - 1) for clip in clips]

    comparison, frames = diff(clips[0], clips[1], thr=thr, return_ranges=True)

    if exclude_ranges:
        frames = [x for x in frames if x not in get_list(exclude_ranges)]

    with Path("./diff.txt").open("a") as log:
        if name:
            log.write(f"{name}=")

        if frames:
            log.write(f"{repr(frames)} \n\n")
        else:
            log.write("no differences found")

    return comparison
