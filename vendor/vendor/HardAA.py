import vapoursynth as vs
import mvsfunc as mvf
import math
core = vs.core

# HardAA for really hard anti-aliasing purposes
# Version: 0.1
# Script's output should be similiar to HiAA, but only "eedi3" and "eedi3+sangnom2" modes ported.
# Some portions of code copied from edi_rpow2.py, thx original author for his efforts on that script.
#
# Requirements:
#     EEDI3 by HollyWu
#     SangNom (if needed)
#     TMaskCleaner (for 'precisecmb' mask)
#     fmtconv (if needed)
#
# Usage:
#     import HardAA
#     clip = HardAA.HardAA(clip, mask='simple', mthr=30, rfactor=2, alpha=0.4, beta=0.2, gamma=15.0, nrad=3, mdis=20, hp=False, vcheck=2, vthreshmul=1, vthresh2=4.0, LumaOnly=True, useCL=False, sangnomPP=False)
#
#     mask: mask mode (simple, simplecmb, precisecmb)
#     mthr: mask threshold (auto-scaled to bitdepth)
#     rfactor: scaling factor for eedi3
#     alpha, beta, gamma, nrad, mdis, hp, vcheck, vthreshmul, vthresh2: eedi3 stuff, refer to it's manual
#     LumaOnly: process only Y plane and copy UV if True
#     useCL: use eedi3CL
#     sangnomPP: use sangnom after eedi3


def HardAA(clip, mask='simple', mthr=30, rfactor=2, alpha=0.4, beta=0.2, gamma=15.0, nrad=3, mdis=20, hp=False, vcheck=2,
           vthreshmul=1, vthresh2=4.0, LumaOnly=True, useCL=False, sangnomPP=False):

    vthresh0 = 32 * vthreshmul
    vthresh1 = 64 * vthreshmul
    if LumaOnly is True:
        iclip = mvf.GetPlane(clip, 0)
        planes = [0]
    else:
        iclip = clip
        planes = [0, 1, 2]
    if useCL is True:
        rpow2 = eedi3cl_rpow2
    else:
        rpow2 = eedi3_rpow2
    aaclip = rpow2(iclip, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, hp=hp, vcheck=vcheck,
                   vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2)
    if sangnomPP is True:
        aaclip = core.sangnom.SangNom(aaclip, order=0, aa=48,
                                      planes=planes).std.Transpose().sangnom.SangNom(order=0, aa=48,
                                                                                     planes=planes).std.Transpose()
    aaclip = core.resize.Spline36(aaclip, clip.width, clip.height, format=iclip.format, src_left=-0.5, src_top=-0.5)
    mask = HardAA_mask(clip, mask=mask, mthr=mthr)
    if LumaOnly is False and mask.format.color_family is not vs.YUV and clip.format.color_family is not vs.GRAY:
        mask_c = core.resize.Spline16(mask, mask.width >> clip.format.subsampling_w,
                                      mask.height >> clip.format.subsampling_h,
                                      src_left=-0.25)                                  # TODO: make zero shift when input is YUV444
        mask = core.std.ShufflePlanes([mask, mask_c, mask_c], planes=[0, 0, 0], colorfamily=clip.format.color_family)
    masked = core.std.MaskedMerge(iclip, aaclip, mask)
    if LumaOnly is True and clip.format.color_family is not vs.GRAY:
        return core.std.ShufflePlanes(clips=[masked, mvf.GetPlane(clip, 1),
                                             mvf.GetPlane(clip, 2)], planes=[0, 0, 0],
                                      colorfamily=clip.format.color_family)
    return masked


def HardAA_mask(clip, mask='simple', mthr=30):

    bits = clip.format.bits_per_sample
    maxvalue = (1 << bits) - 1
    mthr = mthr * maxvalue // 0xFF
    if mask == 'simple':
        mclip = core.std.Prewitt(clip)
        mclip = core.std.Expr(mclip, [f'x {mthr} < 0 {maxvalue} ?']).rgvs.RemoveGrain(17).std.Maximum().std.Inflate()
        mclip = mvf.GetPlane(mclip, 0)
    elif mask == 'simplecmb':
        if clip.format.color_family is not vs.YUV:
            raise ValueError('HardAA mask: simplecmb requires YUV input!')
        mclip = core.std.Prewitt(clip)
        mclip_c = core.std.Expr([mvf.GetPlane(mclip, 1), mvf.GetPlane(mclip, 2)],
                                ['x {ufac} * y {vfac} * max'.format(ufac=1.5, vfac=1.5)])
        mclip_c = core.resize.Point(mclip_c, clip.width, clip.height, src_left=0.25).std.Minimum()
        mclip = core.std.Expr([mvf.GetPlane(mclip, 0), mclip_c], ['x y +'])
        mclip = core.std.Binarize(mclip, mthr).rgvs.RemoveGrain(17).std.Maximum().std.Inflate()
    elif mask == 'precisecmb':
        if clip.format.color_family is not vs.YUV:
            raise ValueError('HardAA mask: precisecmb requires YUV input!')
        mclip = core.std.Sobel(clip)
        mclip_c = core.std.Expr([mvf.GetPlane(mclip, 1), mvf.GetPlane(mclip, 2)],
                                ['x {ufac} * y {vfac} * max'.format(ufac=1.5, vfac=1.5)])
        mclip_c = core.resize.Point(mclip_c, clip.width, clip.height, src_left=0.25).std.Minimum().std.Minimum()
        mclip = core.std.Expr([mvf.GetPlane(mclip, 0), mclip_c], ['x y +'])
        mclip = core.std.Maximum(mclip).std.Minimum().tmc.TMaskCleaner(clip.height / 36, 30 * maxvalue //
                                                                       0xFF).std.Binarize(mthr).std.Inflate()
    else:
        raise ValueError('HardAA mask: unsupported mask!')
    return mclip


def eedi3_rpow2(clip, alpha=None, beta=None, gamma=None, nrad=None, mdis=None, hp=None, ucubic=None, cost3=None,
                vcheck=None, vthresh0=None, vthresh1=None, vthresh2=None):

    def edi(clip, field, dh):
        return core.eedi3m.EEDI3(clip=clip, field=field, dh=dh, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad,
                                 mdis=mdis, hp=hp, ucubic=ucubic, cost3=cost3, vcheck=vcheck, vthresh0=vthresh0,
                                 vthresh1=vthresh1, vthresh2=vthresh2)

    return edi_rpow2(clip=clip, edi=edi)


def eedi3cl_rpow2(clip, alpha=None, beta=None, gamma=None, nrad=None, mdis=None, hp=None, ucubic=None, cost3=None,
                  vcheck=None, vthresh0=None, vthresh1=None, vthresh2=None):

    def edi(clip, field, dh):
        return core.eedi3m.EEDI3CL(clip=clip, field=field, dh=dh, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad,
                                   mdis=mdis, hp=hp, ucubic=ucubic, cost3=cost3, vcheck=vcheck, vthresh0=vthresh0,
                                   vthresh1=vthresh1, vthresh2=vthresh2)

    return edi_rpow2(clip=clip, edi=edi)


def edi_rpow2(clip, edi):

    clip = edi(clip, field=1, dh=1)
    clip = core.std.Transpose(clip)
    clip = edi(clip, field=1, dh=1)
    clip = core.std.Transpose(clip)
    return clip
