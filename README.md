This repo is forked from https://github.com/JDAI-CV/fast-reid and provided some demo-codes based on NNI(https://github.com/microsoft/nni/)

## features
* NNI-HPO , Hyperparameter Optimization
* NNI-NAS, Neural Architecture Search

## TODO
* NNI-MC, Model Compression, Prune

## Start a new HPO exp
Firstly, you should install the nni package (visit https://blog.csdn.net/wb3533366/article/details/134978044 for more details )


`
pip install nni
`

then:

`
python nni_tune.py
`  

## Start a new NAS exp

Multi-GPU is not supported now

`
python nni_nas.py --config-file ./configs/Market1501/sbs_elan_nas.yml --num-gpus 1
`  


## License

Fastreid is released under the [Apache 2.0 license](LICENSE).

## Citing FastReID

If you use FastReID in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.

```BibTeX
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
