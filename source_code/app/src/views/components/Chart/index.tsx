import React, { FC } from 'react';
import { Box, BoxProps, Grid } from '@material-ui/core';
import MDIcon from '@mdi/react';
import {
	BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Brush, Label,
	Tooltip, Legend, LabelList, ResponsiveContainer
} from 'recharts';

interface IProps {
	label?: string
	isError?: boolean
}

const Chart: FC<IProps> = ({ data }) => {
	return (
		<Grid style={{
			height: '100%',
			width: '100%',
			minHeight: '250px',
			maxheight: '500px',
		}}>
			<ResponsiveContainer width="100%" height="100%">
				<BarChart width={400} height={300} data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5, }}>
					<CartesianGrid strokeDasharray="3 3" />
					<XAxis dataKey="group" />
					<YAxis />
					<Tooltip />
					<Legend />
					<Bar dataKey="children" stackId="a" fill="#0088FE" >
						<LabelList dataKey="children" position="middle" />
					</Bar>
					<Bar dataKey="happy_children" stackId="c" fill="#BDE2B9" >
						<LabelList dataKey="happy_children" position="middle" />
					</Bar>
					<Bar dataKey="teenager" stackId="a" fill="#00C49F" >
						<LabelList dataKey="teenager" position="middle" />
					</Bar>
					<Bar dataKey="happy_teenager" stackId="c" fill="#B2B0EA" >
						<LabelList dataKey="happy_teenager" position="middle" />
					</Bar>
					<Bar dataKey="adult" stackId="a" fill="#FFBB28" >
						<LabelList dataKey="adult" position="middle" />
					</Bar>
					<Bar dataKey="happy_adult" stackId="c" fill="#F4B678" >
						<LabelList dataKey="happy_adult" position="middle" />
					</Bar>
					<Bar dataKey="middleAge" stackId="a" fill="#FF8042" >
						<LabelList dataKey="middleAge" position="middle" />
					</Bar>
					<Bar dataKey="happy_middleAge" stackId="c" fill="#8BC1F7" >
						<LabelList dataKey="happy_middleAge" position="middle" />
					</Bar>
					<Bar dataKey="elderly" stackId="a" fill="#red" >
						<LabelList dataKey="elderly" position="middle" />
					</Bar>
					
					<Bar dataKey="happy_elderly" stackId="c" fill="#A2D9D9" >
						<LabelList dataKey="happy_elderly" position="middle" />
					</Bar>

					<Bar dataKey="male" stackId="b" fill="#F0AB00">
						<LabelList dataKey="male" position="middle" />
					</Bar>
					<Bar dataKey="female" stackId="b" fill="#004B95" />
					<Brush height={10} travellerWidth={3} />
				</BarChart>
			</ResponsiveContainer>
		</Grid>
	);
};

// const Chart: FC<IProps> = ({ id, label, isError = false, children }) => {
// 	return (
// 		<Grid style={{
// 			height: '100%',
// 			width: '100%',
// 			minHeight: '250px',
// 			maxheight: '400px',
// 		}}>
// 			<ResponsiveContainer width="100%" height="100%">
// 				<BarChart width={500} height={300} data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5, }}>
// 					<CartesianGrid strokeDasharray="3 3" />
// 					<XAxis dataKey="name" />
// 					<YAxis />
// 					<Tooltip />
// 					<Legend />
// 					<Bar dataKey="children" stackId="a" fill="#7CC674" />
// 					<Bar dataKey="teenager" stackId="a" fill="#8481DD" />
// 					<Bar dataKey="adult" stackId="a" fill="#EF9234" />
// 					<Bar dataKey="middleAge" stackId="a" fill="#06C" />
// 					<Bar dataKey="elderly" stackId="a" fill="#73C5C5" />

// 					<Bar dataKey="male" stackId="b" fill="#F0AB00">
// 						<LabelList dataKey="male" position="middle" />
// 					</Bar>
// 					<Bar dataKey="female" stackId="b" fill="#004B95" />

// 				</BarChart>
// 			</ResponsiveContainer>
// 		</Grid>
// 	);
// };

export default Chart;