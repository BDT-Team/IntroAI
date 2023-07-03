/*global chrome*/
import './App.css';
import {Button, Progress, Switch} from 'antd';
import {motion, animate} from 'framer-motion';
import { useState, useEffect, useRef } from "react";
import axios from 'axios';

function Progressbar({value}) {
    const progressTextRef = useRef(null);
    useEffect(() => {
        const progressText = progressTextRef.current?.textContent;
        if(progressText != null) {
            animate(parseInt(progressText),value, {
                duration: 2,
                onUpdate : (cv) => {
                    progressTextRef.current.textContent = cv.toFixed(0) == null ? "0" : cv.toFixed(0);
                }
            });
        }
    }, [value]);
    return(
        <div className="progressbar-container">
            <div className="progressbar">
                <motion.div 
                className="bar"
                animate={{
                    width: `${value}%`
                }}
                transition={{
                    duration: 2
                }}
                />
            </div>
            <div className="progressbar-text-container">
                <p ref={progressTextRef}>0</p>
                <p>%</p>
            </div>
        </div>
    );
}


function App() {

	const [isRunning, setIsRunning] = useState(false);
	const [percent, setPercent] = useState(0);
	const [positive, setPositive] = useState(0);
	const [neutral, setNeutral] = useState(0);
	const [negative, setNegative] = useState(0);
	const [positiveNoSpam, setPositiveNoSpam] = useState(0);
	const [neutralNoSpam, setNeutralNoSpam] = useState(0);
	const [negativeNoSpam, setNegativeNoSpam] = useState(0);
	const [spam, setSpam] = useState(0);
	const [showResult, setShowResult] = useState(false);
	const [totalComment, setTotalComment] = useState(0);
	const [filterType, setFilterType] = useState(false)

	async function getComment(url) {
		const comments = [];
		const regex = /i\.(\d+)\.(\d+)/;
		const match = url.match(regex);
		let totalRatings;
		let offset = 0;

		if (match) {
			const shop_id = match[1];
			const item_id = match[2];
			const ratings_url = `https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid=${item_id}&limit=20&offset={offset}&shopid=${shop_id}&type=0`;

			const data = await axios.get(ratings_url.replace('{offset}', offset.toString())).then((response) => response.data);
			const no_need_tag_length = data.data.ratings[0].template_tags.length;

			axios.get('https://shopee.co.id/api/v2/item/get_ratings', {
				params: {
				itemid: item_id,
				shopid: shop_id,
				limit: 1,
				offset: 0,
				filter: 6
				}
			})
				.then(response => {
					totalRatings = response.data.data.item_rating_summary.rating_total;
					setTotalComment(totalRatings);
				})
				.catch(error => {
					alert('Error:', error);
			});

			while (true) {

				const data = await axios.get(ratings_url.replace('{offset}', offset.toString()),
				).then((response) => response.data).catch(error => {
					alert(error)
				});

				for (let i = 0; i < data.data.ratings.length; i++) {
					const aggeCommnent = data.data.ratings[i].comment;
					const splitedComments = aggeCommnent.split("\n");
					const n = no_need_tag_length; // Number of elements to remove from the beginning

					splitedComments.splice(0, n);
					const realComment = splitedComments.join("\n");
					comments.push(realComment);
				}
					setPercent(comments.length * 80/ totalRatings)
					if (data.data.ratings.length < 20) {
					break;
				}
				offset += 20;
			}
			} else {
			console.log('No match found.');
		}
	
		return comments;
	}

	const filterTyleChange = () => {
		setFilterType(prev => !prev);
	}

	const process = () => {
		setIsRunning(true);

		chrome.permissions.request({
			origins: ["<all_urls>"]
		});

		chrome.tabs.query({ active: true, currentWindow: true }, async function(tabs) {
			if (tabs && tabs.length > 0) {
			  const currentTab = tabs[0];

			  const url = currentTab.url;

			  const comments = await getComment(url).then(comments => comments);
			  
			  await axios.post('http://127.0.0.1:8000/items', comments, {
				headers: {
					'Content-Type': 'application/json; charset=UTF-8'
			   }
			})
				.then((response) => 
				{
					setPositive(response.data.positive);
					setNeutral(response.data.neutral);
					setNegative(response.data.negative);
					setSpam(response.data.spam)
					setPositiveNoSpam(response.data.positive_ham);
					setNeutralNoSpam(response.data.neutral_ham);
					setNegativeNoSpam(response.data.negative_ham);
					setPercent(90);
					setTimeout(() => {
						setPercent(100);
						setTimeout(() => {
							setShowResult(true);
						}, 2000)
					}, 1000)
				}
				)
				.catch((error) => console.log(error));
			}
		});

	}
	
  return (
    <div className="App">
		{!isRunning ? <div className="extensionTitle">Hust shopee comment filter</div> : ""}
		<div style={{color: '#234869', fontWeight: 700, fontSize: '20px', marginBottom: '20px'}}>{!isRunning ? "Let's see the truth about this product !!" : (!showResult ? "Processing, wait for a while ... " : (negative < 0.08 ? "This product is good. Choose it" : "It seems not too good, see more detail !"))}</div>
		{!isRunning ? <Button style={{background: '#234869', color: '#fff', marginTop: '20px'}} size={"large"} onClick={process}>Run now</Button>
		 : (!showResult ? <Progressbar value={percent} /> : "")}
		 {showResult ? 
		 (
			 <div>
				<div className="filterType">
					<div style={{color: '#3d5c98', fontWeight: 700}}>Remove spam: </div>
					<Switch onChange={filterTyleChange} />
				</div>
				<div className="result">
					<div style={{display: 'flex', gap: "20px", justifyContent: 'center'}}>
						<div className="title">
							<p className="positiveTitle" style={{color: "#234869"}}>Positive: </p>
							<p className="neutralTitle" style={{color: "#234869"}}>Neutral:</p>
						</div>
						<div className="progress">
							<Progress type="circle" size={65} strokeWidth={10} strokeColor="#234869" percent={!filterType ? Math.floor(positive * 100/ totalComment) : Math.floor(positiveNoSpam * 100 / (positiveNoSpam + negativeNoSpam +neutralNoSpam))} />
							<Progress type="circle" size={65} strokeWidth={10} strokeColor="#234869" percent={!filterType ? Math.floor(neutral * 100/ totalComment) : Math.floor(neutralNoSpam * 100 / (positiveNoSpam + negativeNoSpam +neutralNoSpam))} />
						</div>
					</div>
					<div style={{display: 'flex', gap: "20px", justifyContent: 'center'}}>
						<div className="title">
							<p className="negativeTitle" style={{color: "#234869"}}>Negative:</p>
							{!filterType ? <p className="spamTitle" style={{color: "#234869"}}>Spam:</p> : "" }
						</div>
						<div className="progress">
							<Progress type="circle" size={65} strokeWidth={10} strokeColor="#234869" percent={!filterType ? (100 - Math.floor(positive * 100/ totalComment) - Math.floor(neutral * 100/ totalComment)) : (100 - Math.floor(positiveNoSpam * 100 / (positiveNoSpam + negativeNoSpam +neutralNoSpam)) - Math.floor(neutralNoSpam * 100 / (positiveNoSpam + negativeNoSpam +neutralNoSpam)))} />
							{!filterType ? <Progress type="circle" size={65} strokeWidth={10} strokeColor="#234869" percent={Math.floor(spam * 100/ totalComment)} /> : <div></div>}
						</div>
					</div>
				</div>
			</div>
		 )
		: ""}
    </div>
  );
}

export default App;
