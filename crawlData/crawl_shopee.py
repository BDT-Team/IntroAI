from tkinter import E
from xml.etree.ElementTree import Comment
from selenium import webdriver
from time import sleep
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def getCommentsOfEachPage():

	flag = 1
	try:
		WebDriverWait(driver, 10).until(EC.visibility_of_all_elements_located((By.CLASS_NAME, "shopee-product-rating")))
		ratings = driver.find_elements(By.CLASS_NAME, "shopee-product-rating")
	except:
		return []
	
	comments = []

	for rating in ratings:

		rating_content = rating.find_element(By.CLASS_NAME, "shopee-product-rating__main")
		rating_txt =  rating_content.find_element(By.CSS_SELECTOR, "div > div:nth-child(4)")

		if len(rating_txt.find_elements(By.CSS_SELECTOR,"*")) > 0 and rating_txt.get_attribute("class") == 'Rk6V+3':
			comment = rating_txt.find_element(By.CSS_SELECTOR, "div > div:last-child").text
			flag = 0

		elif len(rating_txt.find_elements(By.CSS_SELECTOR,"*")) == 0 and rating_txt.get_attribute("class") == 'Rk6V+3':
			comment = rating_txt.text
			flag = 0

		else :
			comment = ""
		comments.append(comment)
		file.write(comment)
		file.write('\t')

	if flag == 1:
		return []
	else:
		return comments



def crawl_shopee(URL):

	# declare browser
	result = []

	#open URL
	driver.get(URL)
	sleep(2)

	currentPage = driver.find_element(By.CLASS_NAME, "shopee-button-solid")
	nextPageNum = currentPage.find_element(By.XPATH, "following-sibling::*").text

	while nextPageNum.isdigit():

		print("\npage: " + nextPageNum + "\n")

		if len(getCommentsOfEachPage()) == 0:
			break;
		else:
			result = result + getCommentsOfEachPage()
		
		currentPage.find_element(By.XPATH, "following-sibling::*").click()
		currentPage = driver.find_element(By.CLASS_NAME, "shopee-button-solid")
		nextPageNum = currentPage.find_element(By.XPATH, "following-sibling::*").text

driver = webdriver.Chrome('chromefriver.exe')
file = open("comments.txt", "a+")
URL = 'https://shopee.vn/M%C5%A9-bucket-Nhung-T%C4%83m-th%C3%AAu-ch%E1%BB%AF-Promissyou-v%C3%A0nh-n%C3%B3n-tai-b%C3%A8o-tr%C3%B2n-nam-n%E1%BB%AF-th%E1%BB%9Di-trang-SG102-i.86945057.21005603156?sp_atk=ed562f1f-aebd-4696-b7ee-992222e27b3e&xptdk=ed562f1f-aebd-4696-b7ee-992222e27b3e';
crawl_shopee(URL)
file.close()

