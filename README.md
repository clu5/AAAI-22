# Fair Conformal Predictors for Applications in Medical Imaging

> Deep learning has the potential to automate many clinically useful tasks in medical imaging. However translation of deep learning into clinical practice has been hindered by issues such as lack of the transparency and interpretability in these "black box" algorithms compared to traditional statistical methods. Specifically, many clinical deep learning models lack rigorous and robust techniques for conveying certainty (or lack thereof) in their predictions -- ultimately limiting their appeal for extensive use in medical decision-making. Furthermore, numerous demonstrations of algorithmic bias have increased hesitancy towards deployment of deep learning for clinical applications. To this end, we explore how conformal predictions can complement existing deep learning approaches by providing an intuitive way of expressing uncertainty while facilitating greater transparency to clinical users. In this paper, we conduct field interviews with radiologists to assess possible use-cases for conformal predictors. Using insights gathered from these interviews, we devise two clinical use-cases and empirically evaluate several methods of conformal predictions on a dermatology photography dataset for skin lesion classification. We show how to modify conformal predictions to be more adaptive to subgroup differences in patient skin tones through equalized coverage. Finally, we compare conformal prediction against measures of epistemic uncertainty.

Published in AAAI 2022 -- https://ojs.aaai.org/index.php/AAAI/article/view/21459

* [Tutorial on conformal prediction](http://people.eecs.berkeley.edu/~angelopoulos/blog/posts/gentle-intro/)
* [Dataset](https://github.com/mattgroh/fitzpatrick17k)
* [Paper](https://arxiv.org/abs/2109.04392)

Please cite our work as
```
@article{Lu_Lemay_Chang_Höbel_Kalpathy-Cramer_2022, 
title={Fair Conformal Predictors for Applications in Medical Imaging}, 
volume={36}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/21459}, 
DOI={10.1609/aaai.v36i11.21459}, 
number={11}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Lu, Charles and Lemay, Andréanne and Chang, Ken and Höbel, Katharina and Kalpathy-Cramer, Jayashree}, 
year={2022}, 
month={Jun.}, 
pages={12008-12016} 
}
```
