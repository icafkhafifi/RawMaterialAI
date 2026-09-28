# Raw Material AI

## Visual Material & Supply Chain Intelligence

Raw Material AI is a computer-vision prototype that connects visual object detection with material intelligence and upstream supply-chain information.

The current prototype detects bottles, classifies their material as PET or Glass, maps the material to associated raw materials, and displays related companies and public market information.

## Project Overview

The system follows this pipeline:

**Camera → Object Detection → Material Classification → Raw Materials → Related Companies → Market Data**

The goal is to move beyond identifying an object as simply a "bottle" and provide additional material and supply-chain context.

## Features

- Real-time bottle detection using YOLO
- PET vs Glass material classification
- Material confidence score
- Upstream raw-material mapping
- Related company lookup
- Company country and public/private status
- Stock ticker information
- Market price and daily change
- Real-time OpenCV dashboard
- JSON-based material and company database

## Current Material Intelligence

### PET

**Material:** PET — Polyethylene Terephthalate

Associated raw materials:

- PTA — Purified Terephthalic Acid
- MEG — Mono Ethylene Glycol

Related companies in the current prototype:

- PT Indo-Rama Synthetics Tbk
- Reliance Industries Limited

### Glass

**Material:** Glass

The current database maps Glass to soda-lime glass.

Associated raw materials:

- Silica sand
- Soda ash
- Limestone
- Dolomite
- Cullet

Related company in the current prototype:

- Türkiye Şişe ve Cam Fabrikaları A.Ş. (Şişecam)

## System Architecture

```text
CAMERA
   |
   v
OBJECT DETECTION
   |
   v
BOTTLE
   |
   v
MATERIAL CLASSIFICATION
   |
   +--------+--------+
   |                 |
   v                 v
  PET              GLASS
   |                 |
   v                 v
PTA + MEG      SODA-LIME GLASS
                     |
                     v
        SILICA SAND / SODA ASH
        LIMESTONE / DOLOMITE
        CULLET
              |
              v
       RELATED COMPANIES
              |
              v
         MARKET DATA