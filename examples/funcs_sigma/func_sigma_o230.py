def graphfunc(leaf):
    g18745 = leaf[:, 0]
    g18752 = leaf[:, 1]
    g18746 = leaf[:, 2]
    g18856 = (g18752 * g18746 * -1.0)
    g18883 = leaf[:, 3]
    g18923 = (g18856 * g18883)
    g18838 = leaf[:, 4]
    g18866 = (g18838 * g18746 * -1.0)
    g18882 = leaf[:, 5]
    g18936 = (g18866 * g18882)
    g18937 = (g18923 + g18936)
    g18839 = leaf[:, 6]
    g18861 = (g18839 * g18746 * -1.0)
    g18881 = leaf[:, 7]
    g18965 = (g18861 * g18881)
    g18966 = (g18937 + g18965)
    g18840 = leaf[:, 8]
    g18871 = (g18840 * g18746 * -1.0)
    g18759 = leaf[:, 9]
    g18971 = (g18871 * g18759)
    g18972 = (g18966 + g18971)
    g19080 = (g18745 * g18972)
    g18835 = leaf[:, 10]
    g18922 = (g18856 * g18881)
    g18934 = (g18866 * g18759)
    g18935 = (g18922 + g18934)
    g19101 = (g18835 * g18935)
    g19102 = (g19080 + g19101)
    g18834 = leaf[:, 11]
    g18921 = (g18856 * g18882)
    g18938 = (g18866 * g18881)
    g18939 = (g18921 + g18938)
    g18961 = (g18861 * g18759)
    g18962 = (g18939 + g18961)
    g19139 = (g18834 * g18962)
    g19140 = (g19102 + g19139)
    g18836 = leaf[:, 12]
    g18920 = (g18856 * g18759)
    g19167 = (g18836 * g18920)
    g19168 = (g19140 + g19167)
    g18739 = leaf[:, 13]
    g19283 = (g19168 * g18739 * -1.0)
    g19364 = (g19283)
    root0 = g19364
    g18785 = leaf[:, 14]
    g19393 = (g18739 * -1.0 + g18785)
    g19408 = (g18739 * -1.0 + g18785)
    g19423 = (g19393 * g19408)
    g19458 = ((g18739)**2)
    g19493 = (g19423 * -1.0 + g19458 * -1.0)
    g18817 = leaf[:, 15]
    g19516 = (g19493 * g18817)
    g19504 = leaf[:, 16]
    g19554 = (g19516 * g19504)
    g19498 = leaf[:, 17]
    g19518 = (g19493 * g19498)
    g19503 = leaf[:, 18]
    g19567 = (g19518 * g19503)
    g19568 = (g19554 + g19567)
    g19499 = leaf[:, 19]
    g19517 = (g19493 * g19499)
    g19502 = leaf[:, 20]
    g19596 = (g19517 * g19502)
    g19597 = (g19568 + g19596)
    g19500 = leaf[:, 21]
    g19519 = (g19493 * g19500)
    g18818 = leaf[:, 22]
    g19602 = (g19519 * g18818)
    g19603 = (g19597 + g19602)
    g19711 = (g18745 * g19603)
    g19553 = (g19516 * g19502)
    g19565 = (g19518 * g18818)
    g19566 = (g19553 + g19565)
    g19732 = (g18835 * g19566)
    g19733 = (g19711 + g19732)
    g19552 = (g19516 * g19503)
    g19569 = (g19518 * g19502)
    g19570 = (g19552 + g19569)
    g19592 = (g19517 * g18818)
    g19593 = (g19570 + g19592)
    g19770 = (g18834 * g19593)
    g19771 = (g19733 + g19770)
    g19551 = (g19516 * g18818)
    g19798 = (g18836 * g19551)
    g19799 = (g19771 + g19798)
    g19836 = (g19393 * g18739 * -1.0)
    g19871 = (g18739 * -1.0 * g19408)
    g19906 = (g19836 * -1.0 + g19871 * -1.0)
    g19921 = (g19906 * g18817)
    g19959 = (g19921 * g19504)
    g19923 = (g19906 * g19498)
    g19972 = (g19923 * g19503)
    g19973 = (g19959 + g19972)
    g19922 = (g19906 * g19499)
    g20001 = (g19922 * g19502)
    g20002 = (g19973 + g20001)
    g19924 = (g19906 * g19500)
    g20007 = (g19924 * g18818)
    g20008 = (g20002 + g20007)
    g20116 = (g18745 * g20008)
    g19958 = (g19921 * g19502)
    g19970 = (g19923 * g18818)
    g19971 = (g19958 + g19970)
    g20137 = (g18835 * g19971)
    g20138 = (g20116 + g20137)
    g19957 = (g19921 * g19503)
    g19974 = (g19923 * g19502)
    g19975 = (g19957 + g19974)
    g19997 = (g19922 * g18818)
    g19998 = (g19975 + g19997)
    g20175 = (g18834 * g19998)
    g20176 = (g20138 + g20175)
    g19956 = (g19921 * g18818)
    g20203 = (g18836 * g19956)
    g20204 = (g20176 + g20203)
    g20291 = (g19799 + g20204 * -0.5)
    root1 = g20291
    return root0,root1

