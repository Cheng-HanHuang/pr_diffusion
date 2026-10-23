"""Offline Pillow fallback for three N1 letter figures when Matplotlib is absent.

This module has no GPU/model/GT-path IO and does not change numerical
analysis. It produces scientifically equivalent PNG/PDF *draft*
visualizations from N1 aggregate/anatomy/score arrays. Matplotlib
remains the preferred renderer when available.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import math

import numpy as np
from PIL import Image,ImageDraw,ImageFont

WIDTH,HEIGHT=1720,1030
BG=(251,252,254)
INK=(36,46,61)
GREY=(108,119,135)
GRID=(223,229,236)
BLUE=(67,115,177)
ORANGE=(233,151,67)
TEAL=(60,159,151)
RED=(218,87,85)
BASE=(135,145,155)
PLOT=(145,130,1540,865) # left, top, right, bottom

def font(size=23):
    for name in ("DejaVuSans.ttf","LiberationSans-Regular.ttf"):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def canvas(title,subtitle):
    img=Image.new("RGB",(WIDTH,HEIGHT),BG)
    d=ImageDraw.Draw(img)
    d.text((100,38),title,fill=INK,font=font(36))
    d.text((100,91),subtitle,fill=GREY,font=font(21))
    left,top,right,bottom=PLOT
    d.line((left,top,left,bottom),fill=INK,width=3)
    d.line((left,bottom,right,bottom),fill=INK,width=3)
    return img,d

def axes(d,lo,hi,title,ticks=5):
    left,top,right,bottom=PLOT
    if hi<=lo:raise ValueError("invalid axis scale")
    for j in range(ticks+1):
        val=lo+(hi-lo)*j/ticks
        y=bottom-(bottom-top)*j/ticks
        d.line((left,y,right,y),fill=GRID,width=2)
        d.text((left-104,y-12),f"{val:.2f}",fill=GREY,font=font(18))
    d.text((20,140),title,fill=GREY,font=font(21))
    def yy(value):
        return bottom-(float(value)-lo)/(hi-lo)*(bottom-top)
    return yy

def label_x(d,mid,text,baseline=884):
    parts=text.split("\n")
    for i,part in enumerate(parts):
        bb=d.textbbox((0,0),part,font=font(18))
        d.text((mid-(bb[2]-bb[0])/2,baseline+i*24),part,
               fill=INK,font=font(18))

def legend(d,items,y=963):
    x=145
    for text,color in items:
        d.rounded_rectangle((x,y,x+26,y+20),radius=4,fill=color)
        d.text((x+35,y-2),text,fill=INK,font=font(19))
        x+=int(56+len(text)*12)

def output(figdir,name,img):
    figdir=Path(figdir)
    figdir.mkdir(parents=True,exist_ok=True)
    png=figdir/(name+".png");pdf=figdir/(name+".pdf")
    if png.exists() or pdf.exists():raise RuntimeError("STOP_R0N_N1_FIGURE_EXISTS")
    img.save(png,"PNG")
    img.save(pdf,"PDF",resolution=160.0)
    if png.stat().st_size<1000 or pdf.stat().st_size<1000:
        raise RuntimeError("STOP_R0N_N1_PIL_FIGURE_EMPTY")

def anatomy(figdir,rows):
    order=sorted(rows,key=lambda r:(r["cohort"],r["family"]))
    img,d=canvas("N1a | Single-run failure anatomy",
        "Frozen PSNR labels, B22 random versus DEV80 enriched - fraction of runs")
    yy=axes(d,0,1,"Fraction")
    left,top,right,bottom=PLOT
    n=len(order)
    if n==0:raise ValueError("empty anatomy")
    stride=(right-left)/n
    width=min(78,stride*.66)
    pieces=("GOOD","ORBIT_ERROR",
        "NONORBIT_CONSISTENT","NONORBIT_INCONSISTENT")
    colors=(BLUE,ORANGE,TEAL,RED)
    for i,r in enumerate(order):
        center=left+(i+.5)*stride
        current=0.
        for key,color in zip(pieces,colors):
            frac=r[key]/r["n_runs"]
            if frac<0 or current+frac>1+1e-8:
                raise RuntimeError("STOP_R0N_N1_FIGURE_FRACTIONS")
            new=current+frac
            d.rectangle((int(center-width/2),int(yy(new)),
                int(center+width/2),int(yy(current))),fill=color)
            current=new
        short="B22" if r["cohort"]=="B22_random" else "DEV80"
        label_x(d,center,short+"\n"+r["family"])
    legend(d,[(name.replace("NONORBIT_","Nonorbit ").replace("_"," ").title(),c)
            for name,c in zip(pieces,colors)])
    output(figdir,"N1_FIG1_ANATOMY",img)

def cost(figdir,grid,selected):
    img,d=canvas("N1d | Repair success versus UNet forwards",
        "All 60 frozen configurations - open circles; marker size reflects Good25 harm")
    yy=axes(d,0,1,"Orbit-repair success")
    left,top,right,bottom=PLOT
    xx=lambda v:left+(float(v)/256)*(right-left)
    for x in (0,32,64,128,192,256):
        xp=xx(x)
        d.line((xp,bottom,xp,bottom+10),fill=INK,width=2)
        t=str(x)
        bb=d.textbbox((0,0),t,font=font(19))
        d.text((xp-(bb[2]-bb[0])/2,bottom+16),t,fill=GREY,font=font(19))
    for r in grid:
        if r["success"] is None:continue
        color=BLUE if r["cohort"]=="DEV80_enriched" else RED
        size=7+min(21,70*float(r["harm"] or 0))
        x,y=xx(r["forwards"]),yy(r["success"])
        d.ellipse((x-size,y-size,x+size,y+size),outline=color,width=3)
    if selected is not None:
        x=xx(selected["forwards"])
        d.line((x,top,x,bottom),fill=INK,width=3)
        d.text((min(x+15,right-330),top+23),
               "DEV80-selected cost",fill=INK,font=font(19))
    d.text((650,922),"UNet image forwards per candidate",fill=INK,font=font(23))
    legend(d,[("B22 random",RED),("DEV80 enriched",BLUE)])
    output(figdir,"N1_FIG2_COST_SUCCESS",img)

def residual(figdir,records,split,cohort):
    img,d=canvas("N1c | Signed residual distributions",
        "Actual per-run samples - kernel-smoothed violins and observed medians")
    groups=defaultdict(list)
    for r in records:
        key=(cohort(r,split),r["family"])
        groups[key].append(float(r["rho_signed_raw"]))
    order=sorted(groups)
    values=[v for g in order for v in groups[g]]
    if not values:raise ValueError("empty residuals")
    lo=min(0,min(values));hi=max(values)
    if hi==lo:hi=lo+1
    pad=.05*(hi-lo)
    lo-=pad;hi+=pad
    yy=axes(d,lo,hi,"Signed residual")
    left,top,right,bottom=PLOT
    n=len(order)
    stride=(right-left)/n
    levels=np.linspace(lo,hi,72)
    bandwidth=max((hi-lo)/48,1e-8)
    for i,key in enumerate(order):
        sample=np.asarray(groups[key],dtype=float)
        if not np.isfinite(sample).all():
            raise RuntimeError("STOP_R0N_N1_FIGURE_NONFINITE")
        center=left+(i+.5)*stride
        color=BLUE if key[0]=="B22_random" else RED
        # Simple compact Gaussian KDE; bandwidth fixed from global range
        # for *visualization only*; statistical quantiles remain unchanged.
        smooth=np.zeros_like(levels)
        for chunk in np.array_split(sample,max(1,int(math.ceil(len(sample)/120)))):
            smooth+=np.exp(-.5*((levels[:,None]-chunk[None,:])/bandwidth)**2).sum(axis=1)
        max_density=float(smooth.max())
        widths= (smooth/max_density)*(stride*.37) if max_density>0 else smooth
        rightpts=[(center+float(w),yy(v)) for w,v in zip(widths,levels)]
        leftpts=[(center-float(w),yy(v)) for w,v in zip(widths[::-1],levels[::-1])]
        d.polygon(rightpts+leftpts,fill=color)
        med=float(np.median(sample))
        d.line((center-stride*.2,yy(med),center+stride*.2,yy(med)),
               fill=INK,width=4)
        label_x(d,center,("B22" if key[0]=="B22_random" else "DEV80")+"\n"+key[1])
    legend(d,[("B22 random",BLUE),("DEV80 enriched",RED)])
    output(figdir,"N1_FIG3_RESIDUAL",img)

def render(out,n1a,n1c,grid,selection,records,split,cohort):
    figdir=Path(out)/"figures"
    anatomy(figdir,n1a)
    cost(figdir,grid,selection)
    residual(figdir,records,split,cohort)
    return {"renderer":"Pillow CPU fallback (matplotlib unavailable)",
            "fig1":"same frozen stacked label fractions",
            "fig2":"same frozen cost-success/Good25 harm data",
            "fig3":"actual raw residual violin from frozen candidates",
            "formats":["PNG","PDF"],
            "note":"kernel density smoothing affects display only, not numeric evidence"}
