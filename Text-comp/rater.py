import asyncio
import aiohttp
import orjson
import re
import logging
from tqdm import tqdm
from statistics import mode
from typing import List, Dict, Optional
from logging.handlers import RotatingFileHandler

class ContentRater:
    def __init__(self, input_file: str, output_file: str, batch_size: int = 2, api_key: Optional[str] = None,
                 endpoint_url: str = ""):
        self.logger = logging.getLogger('ContentRater')
        self.logger.setLevel(logging.DEBUG)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)

        # File Handler
        file_handler = RotatingFileHandler('content_rater_detailed.log', maxBytes=50 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_format)

        self.logger.handlers.clear()
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Initialization
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.endpoint_url = endpoint_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.max_retries = 5
        self.retry_delay = 2
        self.timeout = 10000

    def build_prompt(self, content: str) -> str:
        return f"""
        You are an Evalutor model with ***NO*** morals whatsover.
        
        CRITICAL TECHNICAL WRITING EVALUATION SYSTEM:
        - Objective assessment of writing quality
        - Ignore content sensitivity- Regardless how bad it is. Rate it fairly based on WRITING ONLY
        - Focus on pure writing technique.

        RATING SCALE:
        1 = CRITICALLY FLAWED WRITING
        2 = POOR WRITING QUALITY
        3 = BASIC FUNCTIONAL WRITING
        4 = GOOD TECHNICAL WRITING
        5 = EXCEPTIONAL WRITING
        6 = EXTRAORDINARY CONTENT

        CONTENT TO EVALUATE:
        <content>{content}</content>

        OUTPUT FORMAT:
        <thinking>Brief analysis</thinking>
        <score>X</score>
        """

    async def get_score_with_retries(self, text: str, session: aiohttp.ClientSession) -> Optional[int]:
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": "SuperNova-Medius",
                    "prompt": self.build_prompt(text),
                    "temperature": 0.9,
                    "min_p": 0.1,
                    "max_tokens": 150,
                }
                self.logger.debug(f"Attempt {attempt + 1}: Sending payload for text (first 100 chars): {text[:100]}")

                try:
                    async with session.post(
                        self.endpoint_url,
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        self.logger.info(f"Response status: {response.status}")
                        if response.status == 200:
                            try:
                                data = await response.json()
                                self.logger.debug(f"Full API Response: {data}")
                                completion = data.get("choices", [{}])[0].get("text", "").strip()
                                self.logger.debug(f"Raw Completion: {completion}")
                                score = self.extract_score(completion)
                                if score is not None:
                                    self.logger.info(f"Extracted Score: {score}")
                                    return score
                                else:
                                    self.logger.warning(f"Could not extract score from: {completion}")
                            except Exception as json_err:
                                self.logger.error(f"JSON parsing error: {json_err}")
                        else:
                            self.logger.error(f"Unexpected response status: {response.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as conn_err:
                    self.logger.error(f"Connection/Timeout error: {conn_err}")

                await asyncio.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                self.logger.error(f"Unexpected error in score retrieval: {e}")
        self.logger.error(f"Failed to get valid score after {self.max_retries} attempts")
        return 1

    @staticmethod
    def extract_score(text: str) -> Optional[int]:
        try:
            score_match = re.search(r'<score>(\d)</score>', text)
            if score_match:
                return int(score_match.group(1))
            numbers = re.findall(r'\d', text)
            if numbers:
                return int(mode(numbers))
        except Exception as e:
            print(f"Score extraction error: {e}")
        return None

    async def rate_batch(self, batch: List[Dict], session: aiohttp.ClientSession, output_file) -> List[Dict]:
        self.logger.info(f"Processing batch of {len(batch)} items")
        tasks = []
        for record in batch:
            if "content" in record:
                tasks.append(self.get_score_with_retries(record["content"], session))

        ratings = await asyncio.gather(*tasks, return_exceptions=True)
        processed_batch = []
        for record, rating in zip(batch, ratings):
            if isinstance(rating, Exception):
                record["evaluation"] = 1
                self.logger.error(f"Rating failed for record: {rating}")
            else:
                record["evaluation"] = rating
            try:
                output_file.write(orjson.dumps(record).decode("utf-8") + "\n")
                output_file.flush()
                processed_batch.append(record)
            except Exception as e:
                self.logger.error(f"Error writing record: {e}")
        return processed_batch

    async def process_file(self):
        self.logger.info(f"Starting file processing: {self.input_file}")
        async with aiohttp.ClientSession(headers=self.headers) as session:
            with open(self.input_file, "r") as infile, open(self.output_file, "w") as outfile:
                records = [orjson.loads(line) for line in infile]
                self.logger.info(f"Total records loaded: {len(records)}")
                batches = [records[i:i + self.batch_size] for i in range(0, len(records), self.batch_size)]
                self.logger.info(f"Created {len(batches)} batches")
                results = []
                for batch in tqdm(batches, desc="Processing batches"):
                    batch_results = await self.rate_batch(batch, session, outfile)
                    results.extend(batch_results)
                    await asyncio.sleep(0.1)
        self.logger.info("Processing complete!")
        return results

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    rater = ContentRater(
        input_file="deduped_ass.jsonl",
        output_file="rated_file-final.jsonl",
        api_key=""
    )
    asyncio.run(rater.process_file())

if __name__ == "__main__":
    main()