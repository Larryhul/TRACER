# Third-party notices

## SAGE, MID and Trajectron++ compatibility

TRACER retains the `sage` package name and selected SAGE modules for import and model compatibility. Its top-level legacy modules derive from [Gutianpei/MID](https://github.com/Gutianpei/MID), whose upstream README declares MIT licensing. The original declaration and author attribution are retained in [MID_README.md](third_party/notices/MID_README.md).

Original MID authors: Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu. The inherited MID checkout identity is recorded in [upstream provenance](provenance/upstream.json).

Trajectron++-derived components retain the [Stanford Autonomous Systems Lab MIT notice](third_party/notices/Trajectron++_LICENSE.txt). Embedded `numpy_ringbuffer`-derived utilities retain the [upstream notice](third_party/notices/numpy_ringbuffer_LICENSE.txt).

## External projects and license scope

SocialGym2, CrowdNav, SICNav/CrowdSimPlus, SACSoN, Python-RVO2 and Bi3 are separate dependencies. Their full snapshots are not redistributed here. See [third_party/README.md](third_party/README.md) for source links and snapshot identity, and follow each project's license and installation requirements.

The repository's [MIT license](LICENSE) applies to original code and repository documentation. Upstream notices remain in force. Manuscripts, figures, IEEE class files and dataset inputs retain their separate rights and notices; the code license does not relicense them.
