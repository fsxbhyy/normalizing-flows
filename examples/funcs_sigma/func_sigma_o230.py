def graphfunc(leaf):
    g18719 = leaf[:, 0]
    g18726 = leaf[:, 1]
    g18720 = leaf[:, 2]
    g18825 = (g18726 * g18720 * -1.0)
    g18843 = leaf[:, 3]
    g18867 = (g18825 * g18843)
    g18811 = leaf[:, 4]
    g18833 = (g18811 * g18720 * -1.0)
    g18842 = leaf[:, 5]
    g18873 = (g18833 * g18842)
    g18874 = (g18867 + g18873)
    g18812 = leaf[:, 6]
    g18829 = (g18812 * g18720 * -1.0)
    g18841 = leaf[:, 7]
    g18893 = (g18829 * g18841)
    g18894 = (g18874 + g18893)
    g18813 = leaf[:, 8]
    g18837 = (g18813 * g18720 * -1.0)
    g18733 = leaf[:, 9]
    g18897 = (g18837 * g18733)
    g18898 = (g18894 + g18897)
    g18956 = (g18719 * g18898)
    g18809 = leaf[:, 10]
    g18866 = (g18825 * g18841)
    g18871 = (g18833 * g18733)
    g18872 = (g18866 + g18871)
    g18968 = (g18809 * g18872)
    g18969 = (g18956 + g18968)
    g18808 = leaf[:, 11]
    g18865 = (g18825 * g18842)
    g18875 = (g18833 * g18841)
    g18876 = (g18865 + g18875)
    g18891 = (g18829 * g18733)
    g18892 = (g18876 + g18891)
    g18992 = (g18808 * g18892)
    g18993 = (g18969 + g18992)
    g18810 = leaf[:, 12]
    g18864 = (g18825 * g18733)
    g19004 = (g18810 * g18864)
    g19005 = (g18993 + g19004)
    g18713 = leaf[:, 13]
    g19066 = (g19005 * g18713 * -1.0)
    g19107 = (g19066)
    root0 = g19107
    g18759 = leaf[:, 14]
    g19126 = (g18713 * -1.0 + g18759)
    g19138 = (g18713 * -1.0 + g18759)
    g19150 = (g19126 * g19138)
    g19174 = ((g18713)**2)
    g19198 = (g19150 * -1.0 + g19174 * -1.0)
    g18791 = leaf[:, 15]
    g19216 = (g19198 * g18791)
    g19207 = leaf[:, 16]
    g19239 = (g19216 * g19207)
    g19202 = leaf[:, 17]
    g19218 = (g19198 * g19202)
    g19206 = leaf[:, 18]
    g19245 = (g19218 * g19206)
    g19246 = (g19239 + g19245)
    g19203 = leaf[:, 19]
    g19217 = (g19198 * g19203)
    g19205 = leaf[:, 20]
    g19265 = (g19217 * g19205)
    g19266 = (g19246 + g19265)
    g19204 = leaf[:, 21]
    g19219 = (g19198 * g19204)
    g18792 = leaf[:, 22]
    g19269 = (g19219 * g18792)
    g19270 = (g19266 + g19269)
    g19328 = (g18719 * g19270)
    g19238 = (g19216 * g19205)
    g19243 = (g19218 * g18792)
    g19244 = (g19238 + g19243)
    g19340 = (g18809 * g19244)
    g19341 = (g19328 + g19340)
    g19237 = (g19216 * g19206)
    g19247 = (g19218 * g19205)
    g19248 = (g19237 + g19247)
    g19263 = (g19217 * g18792)
    g19264 = (g19248 + g19263)
    g19364 = (g18808 * g19264)
    g19365 = (g19341 + g19364)
    g19236 = (g19216 * g18792)
    g19376 = (g18810 * g19236)
    g19377 = (g19365 + g19376)
    g19392 = (g19126 * g18713 * -1.0)
    g19416 = (g18713 * -1.0 * g19138)
    g19440 = (g19392 * -1.0 + g19416 * -1.0)
    g19452 = (g19440 * g18791)
    g19475 = (g19452 * g19207)
    g19454 = (g19440 * g19202)
    g19481 = (g19454 * g19206)
    g19482 = (g19475 + g19481)
    g19453 = (g19440 * g19203)
    g19501 = (g19453 * g19205)
    g19502 = (g19482 + g19501)
    g19455 = (g19440 * g19204)
    g19505 = (g19455 * g18792)
    g19506 = (g19502 + g19505)
    g19564 = (g18719 * g19506)
    g19474 = (g19452 * g19205)
    g19479 = (g19454 * g18792)
    g19480 = (g19474 + g19479)
    g19576 = (g18809 * g19480)
    g19577 = (g19564 + g19576)
    g19473 = (g19452 * g19206)
    g19483 = (g19454 * g19205)
    g19484 = (g19473 + g19483)
    g19499 = (g19453 * g18792)
    g19500 = (g19484 + g19499)
    g19600 = (g18808 * g19500)
    g19601 = (g19577 + g19600)
    g19472 = (g19452 * g18792)
    g19612 = (g18810 * g19472)
    g19613 = (g19601 + g19612)
    g19660 = (g19377 + g19613 * -0.5)
    root1 = g19660
    return root0,root1

