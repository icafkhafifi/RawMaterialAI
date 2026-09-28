\# Raw Material AI



\## Visual Material \& Supply Chain Intelligence



Raw Material AI is a computer-vision prototype that identifies the material of a detected bottle and connects the material to upstream raw materials, related companies, and public market information.



The current prototype focuses on distinguishing between \*\*PET\*\* and \*\*Glass\*\* bottles using a custom image-classification model.



\---



\## Project Overview



The idea behind Raw Material AI is to connect the physical world with material and supply-chain intelligence.



Instead of only identifying an object as a "bottle", the system attempts to answer:



> What material is this object made from, what raw materials are associated with that material, and which companies are related to it?



The current pipeline is:



Camera → Object Detection → Material Classification → Raw Materials → Related Companies → Market Data



\---



\## Features



\- Real-time bottle detection using YOLO

\- Custom PET vs Glass material classification

\- Material confidence score

\- Raw-material mapping

\- Multiple related company lookup

\- Company country and public/private status

\- Stock ticker information

\- Market price and daily change

\- Real-time OpenCV dashboard

\- JSON-based material and company database



\---



\## Current Material Intelligence



\### PET



Detected material:



\*\*PET — Polyethylene Terephthalate\*\*



Associated raw materials:



\- PTA — Purified Terephthalic Acid

\- MEG — Mono Ethylene Glycol



Related companies in the current prototype:



\- PT Indo-Rama Synthetics Tbk

\- Reliance Industries Limited



\---



\### Glass



Detected material:



\*\*Glass\*\*



The current database maps Glass to soda-lime glass.



Associated raw materials:



\- Silica sand

\- Soda ash

\- Limestone

\- Dolomite

\- Cullet



Related company in the current prototype:



\- Türkiye Şişe ve Cam Fabrikaları A.Ş. (Şişecam)



\---



\## System Architecture



```text

&#x20;                CAMERA

&#x20;                   |

&#x20;                   v

&#x20;         +-------------------+

&#x20;         | Object Detection  |

&#x20;         |      YOLO         |

&#x20;         +-------------------+

&#x20;                   |

&#x20;                   v

&#x20;                BOTTLE

&#x20;                   |

&#x20;                   v

&#x20;      +-------------------------+

&#x20;      | Material Classification |

&#x20;      |       Custom Model      |

&#x20;      +-------------------------+

&#x20;             /          \\

&#x20;            /            \\

&#x20;           v              v

&#x20;         PET            GLASS

&#x20;          |                |

&#x20;          v                v

&#x20;     PTA + MEG       Soda-lime Glass

&#x20;                           |

&#x20;                           v

&#x20;             Silica Sand / Soda Ash

&#x20;             Limestone / Dolomite

&#x20;             Cullet

&#x20;            \\                /

&#x20;             \\              /

&#x20;              v            v

&#x20;            RELATED COMPANIES

&#x20;                    |

&#x20;                    v

&#x20;              MARKET DATA

