"""
A script to download data for forecasted unemployment.

"""

from datetime import datetime as dt
import requests
from bs4 import BeautifulSoup
import chromedriver_binary
from selenium import webdriver
from kora.selenium import wd


def download_pdf(url, name, path):
    r = requests.get(url, allow_redirects=True)
    file_name = url[url.rfind('/')+1:]

    if r.status_code != 404:
        open(path + file_name, 'wb').write(r.content)
    else:
        print('Failed to download the report for ' + name)


def mufj(path):
    # MUFJ
    downloaded = False
    tdy = dt.today()
    start = "https://www.murc.jp/wp-content/uploads/" + str(tdy.year) + "/"

    for i in range(tdy.month, 0, -1):
        file_name = "short_" + str(tdy.year % 100) + str(i).zfill(2) + ".pdf"
        MUFJ_url = start + str(i).zfill(2) + "/" + file_name
        r = requests.get(MUFJ_url, allow_redirects=True)
        if r.status_code != 404:
            open(path + file_name, 'wb').write(r.content)
            downloaded = True
            break

    start = "https://www.murc.jp/wp-content/uploads/" + str(tdy.year-1) + "/"
    for i in range(12, 0, -1):
        year_short = str(tdy.year % 100 - 1)
        file_name = "short_" + year_short + str(i).zfill(2) + ".pdf"
        MUFJ_url = start + str(i).zfill(2) + "/" + file_name
        r = requests.get(MUFJ_url, allow_redirects=True)
        if r.status_code != 404:
            open(path + file_name, 'wb').write(r.content)
            break


def daiwa(path):
    # Daiwa
    daiwa_url = "https://www.dir.co.jp/report/research/economics/outlook/index.html"
    daiwa_html = requests.get(daiwa_url)
    soup = BeautifulSoup(daiwa_html.content, "html.parser")

    for reports in soup.find(class_="mod-reportlist").find_all("a"):
        if "日本経済見通し" in reports.text:
            url = "https://www.dir.co.jp" + reports.get("href")
            url = url[:-4] + "pdf"
            download_pdf(url, "Daiwa", path)
            break


def jri(path):
    # JRI
    jri_url = "https://www.jri.co.jp/report/medium/japan/"
    jri_html = requests.get(jri_url)
    soup = BeautifulSoup(jri_html.content, "html.parser")
    res = soup.find(class_="link-tail").find_all("a")[0].get("href")
    url = "https://www.jri.co.jp" + res
    download_pdf(url, "JRI", path)


def nochuri(path):
    nochuri_url = "https://www.nochuri.co.jp/report/list.html?f_first_realm=3&f_first_realm_detail=14"
    nochuri_html = requests.get(nochuri_url)
    soup = BeautifulSoup(nochuri_html.content, "html.parser")
    for reports in soup.find_all(class_="title"):
        if "経済見通し" in reports.text:
            url = reports.find("a").get("href")
            download_pdf(url, "Nochuri", path)
            break


def scbri(path):
    # Scbri
    scbri_url = "https://www.scbri.jp/keizaimitousi.htm"
    scbri_html = requests.get(scbri_url)
    soup = BeautifulSoup(scbri_html.content, "html.parser")
    res = (soup.find("table").find_all("a")[0]).get("href")
    url = "https://www.scbri.jp/" + res
    download_pdf(url, "SCBRI", path)


def nli(path):
    url = "https://www.nli-research.co.jp/report_tag/tag_id=79?site=nli"
    wd.get(url)
    soup = BeautifulSoup(wd.page_source)
    for report in soup.find_all(class_="item"):
        if"年度経済見通し" in report.find("a").text:
            url2 = (report.find("a")).get("href")[:-9]
            url3 = url2[url2.rfind("id=")+3:]
            start = "https://www.nli-research.co.jp/files/topics/"
            durl = start + url3 + "_ext_18_0.pdf"
            download_pdf(durl, "Nli", path)
            break

    for report in soup.find_all(class_="item"):
        if"中期経済見通し" in report.find("a").text:
            url2 = (report.find("a")).get("href")[:-9]
            url3 = url2[url2.rfind("id=")+3:]
            start = "https://www.nli-research.co.jp/files/topics/"
            durl = start + url3 + "_ext_18_0.pdf"
            download_pdf(durl, "Nli", path)
            break


def hama_ginko(path):
    url = "https://www.yokohama-ri.co.jp/html/report/index.html"
    wd.get(url)
    source = wd.page_source
    soup = BeautifulSoup(source)

    for report in soup.find_all(class_="modal-content"):
        found = False
        class_ = "modal_report_main_content_ttl"
        if "景気予測" in report.find(class_=class_).text:
            for report_a in report.find_all("a"):
                if report_a.get("href")[:3] == "pdf":
                    res = report_a.get("href")
                    url2 = "https://www.yokohama-ri.co.jp/html/report/" + res
                    download_pdf(url2, "Hama", path)
                    found = True
                    break
            if found:
                break


def shinsei(path):
    wd.get("https://www.shinseibank.com/institutional/markets_info/past/")
    source = wd.page_source
    soup = BeautifulSoup(source)
    for report in soup.find(id="monthlyReportList").find_all(class_="pdf"):
        if "日本経済見通し" in report.text:
            download_pdf(report.get("href"), "Shinsei", path)
            break


def nomura(path):
    nomura_url = "https://www.nomuraholdings.com/jp/services/zaikai/"
    nomura_html = requests.get(nomura_url)
    soup = BeautifulSoup(nomura_html.content, "html.parser")
    for report in soup.find_all(class_="c-List-info__link"):
        if "経済見通し" in report.text:
            url = "https://www.nomuraholdings.com" + report.find("a").get("href")
            url = url[:-4] + "pdf"
            start = url[:url.find("journal/")-1]
            end = url[url.find("journal/")+7:]
            url2 = start + "/journal/pdf" + end
            download_pdf(url2, "Nomura", path)
            break


def jcer(path):
    url = "https://www.jcer.or.jp/economic-forecast/short-term"
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    url2 = "https://www.jcer.or.jp/" + soup.find(class_="more").get("href")
    html = requests.get(url2)
    soup = BeautifulSoup(html.content, "html.parser")
    res = soup.find(class_="sec-download section").find("a").get("href")
    url3 = "https://www.jcer.or.jp" + res

    r = requests.get(url, allow_redirects=True)
    file_name = url[url.rfind('/')+1:]
    if r.status_code != 404:
        open(path + 'jcer.pdf', 'wb').write(r.content)
    else:
        print('Failed to download the report for JCER.')


def smtb(path):
    url = "https://www.smtb.jp/others/report/economy/"
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    for report in soup.find_all(class_="linkType_pdf"):
        found = False
        for report_a in report.find_all("a"):
            if "日本経済見通し" in report_a.text:
                url2 = url+report_a.get("href")
                download_pdf(url2, "SMTB", path)
                found = True
                break
        if found:
            break


def tdb(path):
    url = "https://www.tdb-di.com/economic-report/cat3/"
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    for report in soup.find_all(class_="newsList__title"):
        url2 = report.find("a").get("href")
        url3 = "https://www.tdb-di.com/" + url2[len("https://www.tdb-di.com/posts/"):]
        url3 = url3[:-3] + "pdf"
        download_pdf(url3, "TDB", path)
        break


def forecast(params, output_path):
    analysis_date = params['analysis_date']
    path = output_path + analysis_date + '/forecast/'

    mufj(path)
    daiwa(path)
    jri(path)
    nochuri(path)
    scbri(path)
    nli(path)
    hama_ginko(path)
    shinsei(path)
    nomura(path)
    jcer(path)
    smtb(path)
    tdb(path)


if __name__ == '__main__':
    import yaml

    params_path = '../parameters.yml'
    output_path = "../raw_data/"

    # Load parameters
    with open(params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    forecast(params, output_path)
