# Dataset provenance — decision record (M0, provisional)

**Candidate:** VisA — Visual Anomaly, Amazon Science / Zou et al. (ECCV 2022).

Publisher repository: https://github.com/amazon-science/spot-diff

AWS Open Data registry: https://registry.opendata.aws/visa/

Publisher says dataset released under CC BY 4.0; verify exact files and attribution before distribution. There are 10,821 images across 12 categories; PCB1 has 1,004 normal and 100 anomalous images. We have not downloaded or inspected source files yet.

Official one-class setup (`split_csv/1cls.csv`): normal-image training split and normal/anomalous test split with masks. Preserve official test partition; derive a separate validation partition only from allowed training/development data, recording its methodology and hashes. Do not tune a model or choose thresholds using the held-out test.

**Rights checklist:** Dataset license != model weights license != dependencies license. Audit exact pretrained files and restrictions independently. Attribution, notices, and derivative obligations must be met. No proprietary Tesla data or unpublished internal results.

**Not yet decided:** specific PCB category, precise validation method, distribution method for data files, backbone and pretrained weights, final hosting runtime.
