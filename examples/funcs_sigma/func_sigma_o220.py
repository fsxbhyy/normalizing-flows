def graphfunc(leaf):
    g18745 = leaf[:, 0]
    g18752 = leaf[:, 1]
    g18746 = leaf[:, 2]
    g18856 = (g18752 * g18746 * -1.0)
    g18882 = leaf[:, 3]
    g18921 = (g18856 * g18882)
    g18838 = leaf[:, 4]
    g18866 = (g18838 * g18746 * -1.0)
    g18881 = leaf[:, 5]
    g18938 = (g18866 * g18881)
    g18939 = (g18921 + g18938)
    g18839 = leaf[:, 6]
    g18861 = (g18839 * g18746 * -1.0)
    g18759 = leaf[:, 7]
    g18961 = (g18861 * g18759)
    g18962 = (g18939 + g18961)
    g19078 = (g18745 * g18962)
    g18835 = leaf[:, 8]
    g18920 = (g18856 * g18759)
    g19097 = (g18835 * g18920)
    g19098 = (g19078 + g19097)
    g18834 = leaf[:, 9]
    g18922 = (g18856 * g18881)
    g18934 = (g18866 * g18759)
    g18935 = (g18922 + g18934)
    g19131 = (g18834 * g18935)
    g19132 = (g19098 + g19131)
    g18739 = leaf[:, 10]
    g19270 = (g19132 * g18739 * -1.0)
    g19362 = (g19270)
    root0 = g19362
    g18785 = leaf[:, 11]
    g19393 = (g18739 * -1.0 + g18785)
    g19408 = (g18739 * -1.0 + g18785)
    g19423 = (g19393 * g19408)
    g19458 = ((g18739)**2)
    g19493 = (g19423 * -1.0 + g19458 * -1.0)
    g18817 = leaf[:, 12]
    g19516 = (g19493 * g18817)
    g19503 = leaf[:, 13]
    g19552 = (g19516 * g19503)
    g19498 = leaf[:, 14]
    g19518 = (g19493 * g19498)
    g19502 = leaf[:, 15]
    g19569 = (g19518 * g19502)
    g19570 = (g19552 + g19569)
    g19499 = leaf[:, 16]
    g19517 = (g19493 * g19499)
    g18818 = leaf[:, 17]
    g19592 = (g19517 * g18818)
    g19593 = (g19570 + g19592)
    g19709 = (g18745 * g19593)
    g19551 = (g19516 * g18818)
    g19728 = (g18835 * g19551)
    g19729 = (g19709 + g19728)
    g19553 = (g19516 * g19502)
    g19565 = (g19518 * g18818)
    g19566 = (g19553 + g19565)
    g19762 = (g18834 * g19566)
    g19763 = (g19729 + g19762)
    g19836 = (g19393 * g18739 * -1.0)
    g19871 = (g18739 * -1.0 * g19408)
    g19906 = (g19836 * -1.0 + g19871 * -1.0)
    g19921 = (g19906 * g18817)
    g19957 = (g19921 * g19503)
    g19923 = (g19906 * g19498)
    g19974 = (g19923 * g19502)
    g19975 = (g19957 + g19974)
    g19922 = (g19906 * g19499)
    g19997 = (g19922 * g18818)
    g19998 = (g19975 + g19997)
    g20114 = (g18745 * g19998)
    g19956 = (g19921 * g18818)
    g20133 = (g18835 * g19956)
    g20134 = (g20114 + g20133)
    g19958 = (g19921 * g19502)
    g19970 = (g19923 * g18818)
    g19971 = (g19958 + g19970)
    g20167 = (g18834 * g19971)
    g20168 = (g20134 + g20167)
    g20289 = (g19763 + g20168 * -0.5)
    root1 = g20289
    return root0,root1

