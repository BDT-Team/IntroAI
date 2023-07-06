import axios from 'axios';
import { comment } from 'postcss';

async function getComment() {

	const comments = [];
	const url = 'https://shopee.vn/%C4%90%E1%BB%93ng-H%E1%BB%93-%C4%90i%E1%BB%87n-T%E1%BB%AD-D%C3%A2y-Silicone-Th%E1%BB%83-Thao-Cho-H%E1%BB%8Dc-Sinh-i.645432503.16926479707?sp_atk=71087ce0-aec6-4a70-a44f-f6595987f8a4&xptdk=71087ce0-aec6-4a70-a44f-f6595987f8a4';
	const regex = /i\.(\d+)\.(\d+)/;
	const match = url.match(regex);

		if (match) {
		const shop_id = match[1];
		const item_id = match[2];
		const ratings_url = `https://shopee.co.id/api/v2/item/get_ratings?filter=0&flag=1&itemid=${item_id}&limit=20&offset={offset}&shopid=${shop_id}&type=0`;

		let offset = 0;

		const template_tags = await axios.get(ratings_url.replace('{offset}', offset.toString())).then((response) => response.data.data.ratings[0].template_tags);

		axios.get('https://shopee.co.id/api/v2/item/get_ratings', {
			params: {
			  itemid: item_id,
			  shopid: shop_id, // Replace with the actual shop ID
			  limit: 1,
			  offset: 0,
			  filter: 6
			}
		  })
			.then(response => {
			  const commentCount = response.data.data.item_rating_summary;
			  console.log(commentCount);
			})
			.catch(error => {
			  console.error('Error:', error);
			});

		const no_need_tag_length = template_tags.length;

		while (true) {
			const data = await axios.get(ratings_url.replace('{offset}', offset.toString())).then((response) => response.data);

			for (let i = 0; i < data.data.ratings.length; i++) {
				const aggeCommnent = data.data.ratings[i].comment;
				const splitedComments = aggeCommnent.split("\n");
				const n = no_need_tag_length; // Number of elements to remove from the beginning

				splitedComments.splice(0, n);
				const realComment = splitedComments.join("\n");
				if(realComment.length > 0) {
					comments.push(realComment);
				}
			}

				if (data.data.ratings.length < 20) {
				break;
			}

			offset += 20;
		}
		} else {
		console.log('No match found.');
	}
	console.log('end.');

	return comments;
}
const comments = await getComment().then(comments => comments);
console.log("check: ", comments.length);
await axios.post('http://127.0.0.1:8000/items', comments, {
  headers: {
    'Content-Type': 'application/json; charset=UTF-8'
  }
})
  .then((response) => console.log(response.data))
  .catch((error) => console.log(error));

