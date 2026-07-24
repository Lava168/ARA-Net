# Regulatory Notes

This document is a research-planning note, not legal advice.

## Current Status

ARA-Net V6 is released as an open-source research prototype. It is not marketed or intended as a clinical diagnostic device, and it is not cleared or approved for clinical use.

## Why The Boundary Matters

The FDA describes Software as a Medical Device (SaMD) as software intended for one or more medical purposes that performs those purposes without being part of a hardware medical device. FDA materials also describe AI/ML-enabled medical devices and emphasize that medical-device software may require premarket review depending on intended use, risk, and modifications.

Because ARA-Net is an AD staging model using MRI/clinical information, any future claim that it provides diagnostic recommendations for patient care would require formal regulatory assessment before clinical deployment.

## Practical Translation Path

1. Keep this repository as a research prototype.
2. Run retrospective external validation across multiple institutions.
3. Run prospective silent-mode validation.
4. Evaluate human-AI team performance under clinically relevant conditions.
5. Add quality-management, cybersecurity, monitoring, and change-control processes.
6. Seek formal regulatory advice before clinical claims or deployment.

## Official References

- FDA Clinical Decision Support Software guidance: https://www.fda.gov/media/162880/download
- FDA Software as a Medical Device overview: https://www.fda.gov/MedicalDevices/DigitalHealth/SoftwareasaMedicalDevice/default.htm
- FDA AI/ML Software as a Medical Device page: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device
- FDA Good Machine Learning Practice guiding principles: https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- FDA Transparency for Machine Learning-Enabled Medical Devices guiding principles: https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles
