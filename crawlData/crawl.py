from xml.etree.ElementTree import Comment
import numpy as np
from selenium import webdriver
from time import sleep
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By
import random
import pandas as pd

def getCommentsOfEachPage():
	flag = 1
	ratings = driver.find_elements(By.CLASS_NAME, "shopee-product-rating")
	comments = []

	for rating in ratings:
		rating_content = rating.find_element(By.CLASS_NAME, "shopee-product-rating__main")
		rating_txt =  rating_content.find_element(By.CSS_SELECTOR, "div > div:nth-child(4)")
		if len(rating_txt.find_elements(By.CSS_SELECTOR,"*")) == 0 and rating_txt.get_attribute("class") == 'Rk6V+3':
			comment = rating_txt.text
			flag = 0
		elif len(rating_txt.find_elements(By.CSS_SELECTOR,"*")) > 0 and rating_txt.get_attribute("class") == 'Rk6V+3':
			comment = rating_txt.find_element(By.CSS_SELECTOR, "div > div:last-child").text
			flag = 0
		else :
			comment = ""
		comments.append(comment)
	if flag == 1:
		return []
	else:
		return comments


result = []
# declare browser
driver = webdriver.Chrome('chromefriver.exe');

#Open URL
URL = 'https://shopee.vn/-HSD-T8-2023-Ng%C5%A9-c%E1%BB%91c-Calbee-%C4%83n-ki%C3%AAng-gi%E1%BA%A3m-c%C3%A2n-Nh%E1%BA%ADt-B%E1%BA%A3n-v%E1%BB%9Bi-%C4%91%E1%BB%A7-v%E1%BB%8B-ngon-tuy%E1%BB%87t-mix-hoa-qu%E1%BA%A3-tr%C3%A1i-c%C3%A2y-s%E1%BB%AFa-chua-d%C3%B9ng-%C4%83n-s%C3%A1ng-i.7884912.842674750?sp_atk=2e5fd91a-2870-4f2f-a448-ce0b5b0b80d6&xptdk=2e5fd91a-2870-4f2f-a448-ce0b5b0b80d6'; 
driver.get(URL)
sleep(6)
currentPage = driver.find_element(By.CLASS_NAME, "shopee-button-solid")
nextPageNum = currentPage.find_element(By.XPATH, "following-sibling::*").text

while nextPageNum.isdigit():
	print("\npage: " + nextPageNum + "\n")
	if len(getCommentsOfEachPage()) == 0:
		break;
	else:
		result = result + getCommentsOfEachPage()
	currentPage.find_element(By.XPATH, "following-sibling::*").click()
	sleep(0.5)
	currentPage = driver.find_element(By.CLASS_NAME, "shopee-button-solid")
	nextPageNum = currentPage.find_element(By.XPATH, "following-sibling::*").text

for comment in result:
	print(comment)


