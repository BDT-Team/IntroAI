from xml.etree.ElementTree import Comment
from selenium import webdriver
from time import sleep
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def load_url_selenium_lazada(url):
    driver=webdriver.Chrome(executable_path='/usr/bin/chromedriver')
    driver.get(url)
    list_review = []
    x=0
    while x<10:
        try:
            #Get the review details here
            WebDriverWait(driver,5).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR,"div.item")))
        except:
            print('No has comment')
            break
        
        product_reviews = driver.find_elements_by_css_selector("[class='item']")
        # Get product review
        for product in product_reviews:
            review = product.find_element_by_css_selector("[class='content']").text
            if (review != "" or review.strip()):
                print(review, "\n")
                list_review.append(review)
        #Check for button next-pagination-item have disable attribute then jump from loop else click on the next button
        if len(driver.find_elements_by_css_selector("button.next-pagination-item.next[disabled]"))>0:
            break;
        else:
            button_next=WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "button.next-pagination-item.next")))
            driver.execute_script("arguments[0].click();", button_next)
            print("next page")
            time.sleep(2)
            x +=1
    driver.close()
    return list_review


load_url_selenium_lazada('https://www.lazada.vn/products/chi-44-64-hoan-tien-15vay-dam-nu-body-om-dang-khoet-lung-nhun-hong-cuc-xinh-va-quyen-ru-i2049926619-s9569173603.html?spm=a2o4n.home.flashSale.3.19053bdcM3cAzb&search=1&mp=1&c=fs&clickTrackInfo=rs%3A0.05076918751001358%3Bfs_item_discount_price%3A109000%3Bitem_id%3A2049926619%3Bmt%3Ahot%3Bfs_utdid%3A-1%3Bfs_item_sold_cnt%3A1%3Babid%3A287818%3Bfs_item_price%3A260000%3Bpvid%3A5053bd11-2019-46a0-b15b-dce4c5b6147b%3Bfs_min_price_l30d%3A0%3Bdata_type%3Aflashsale%3Bfs_pvid%3A5053bd11-2019-46a0-b15b-dce4c5b6147b%3Btime%3A1680725376%3Bfs_biz_type%3Afs%3Bscm%3A1007.17760.287818.%3Bchannel_id%3A0000%3Bfs_item_discount%3A58%25%3Bcampaign_id%3A217108&scm=1007.17760.287818.0')